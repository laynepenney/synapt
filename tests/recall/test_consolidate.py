"""Tests for memory consolidation — clustering, prompt building, and action application."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synapt.recall.journal import JournalEntry
from synapt.recall.knowledge import KnowledgeNode, append_node, read_nodes, compute_confidence
from synapt.recall.consolidate import (
    CONTEXT_BUDGET,
    CONSOLIDATION_PROMPT_MINIMAL,
    ConsolidationResult,
    EXTRACTION_CAPABILITIES,
    MIN_RESPONSE_TOKENS,
    _DEFAULT_GOOD_EXAMPLES,
    _apply_consolidation_result,
    _build_consolidation_prompt,
    _build_extraction_packet,
    _build_few_shot_examples,
    _cluster_cache_key,
    _dedup_decisions_path,
    _earliest_temporal_ref_date,
    _estimate_response_budget,
    _extract_keywords,
    _extract_source_id,
    _format_existing_knowledge,
    _format_journal_cluster,
    _get_project_context,
    _is_garbled_content,
    _is_generic_node,
    _knowledge_nodes_from_extraction,
    _lacks_specificity,
    _load_response_cache,
    _save_cached_response,
    _jaccard,
    _log_dedup_decision,
    _map_extraction_category,
    _parse_llm_response,
    _split_large_cluster,
    _temporal_window_clusters,
    cluster_journal_entries,
    consolidate,
)


def _make_entry(
    session_id: str = "sess-A",
    timestamp: str = "2026-03-01T00:00:00",
    focus: str = "",
    done: list[str] | None = None,
    decisions: list[str] | None = None,
    next_steps: list[str] | None = None,
    files_modified: list[str] | None = None,
) -> JournalEntry:
    return JournalEntry(
        timestamp=timestamp,
        session_id=session_id,
        focus=focus,
        done=done or [],
        decisions=decisions or [],
        next_steps=next_steps or [],
        files_modified=files_modified or [],
        enriched=True,
    )


class TestExtractKeywords(unittest.TestCase):
    def test_removes_stopwords(self):
        kw = _extract_keywords("the quick brown fox is running fast")
        self.assertNotIn("the", kw)
        self.assertNotIn("is", kw)
        self.assertIn("quick", kw)
        self.assertIn("brown", kw)

    def test_removes_short_words(self):
        kw = _extract_keywords("go to the db")
        # "go" and "to" and "db" are <= 2 chars
        self.assertNotIn("go", kw)
        self.assertNotIn("to", kw)
        self.assertNotIn("db", kw)

    def test_lowercases(self):
        kw = _extract_keywords("SwiftSyntax Parser")
        self.assertIn("swiftsyntax", kw)
        self.assertIn("parser", kw)


class TestJaccard(unittest.TestCase):
    def test_identical_sets(self):
        self.assertAlmostEqual(_jaccard({"a", "b"}, {"a", "b"}), 1.0)

    def test_disjoint_sets(self):
        self.assertAlmostEqual(_jaccard({"a"}, {"b"}), 0.0)

    def test_empty_sets(self):
        self.assertAlmostEqual(_jaccard(set(), set()), 0.0)

    def test_partial_overlap(self):
        j = _jaccard({"a", "b", "c"}, {"b", "c", "d"})
        # intersection=2, union=4 → 0.5
        self.assertAlmostEqual(j, 0.5)


class TestClusterJournalEntries(unittest.TestCase):
    def test_cluster_by_file_overlap(self):
        e1 = _make_entry(
            session_id="s1",
            files_modified=["src/foo.py", "src/bar.py", "src/baz.py"],
            focus="Refactored foo module",
        )
        e2 = _make_entry(
            session_id="s2",
            files_modified=["src/foo.py", "src/bar.py"],
            focus="Fixed bug in foo",
        )
        clusters = cluster_journal_entries([e1, e2])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), 2)

    def test_cluster_by_keyword_overlap(self):
        e1 = _make_entry(
            session_id="s1",
            focus="Training adapters with MLX locally",
            decisions=["Use adapter distillation pipeline"],
        )
        e2 = _make_entry(
            session_id="s2",
            focus="Adapter training on Modal cloud",
            decisions=["Use adapter checkpoints"],
        )
        # Both share keywords: "adapter", "training"
        clusters = cluster_journal_entries([e1, e2])
        self.assertEqual(len(clusters), 1)

    def test_no_overlap_falls_back_to_temporal(self):
        e1 = _make_entry(
            session_id="s1",
            timestamp="2026-03-01T10:00:00",
            focus="Database migration",
            files_modified=["src/db.py"],
        )
        e2 = _make_entry(
            session_id="s2",
            timestamp="2026-03-01T11:00:00",
            focus="Frontend styling",
            files_modified=["src/ui.css"],
        )
        clusters = cluster_journal_entries([e1, e2])
        # No file overlap, no keyword overlap → temporal fallback
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), 2)

    def test_fewer_than_two_entries(self):
        e1 = _make_entry(session_id="s1", focus="Solo session")
        self.assertEqual(cluster_journal_entries([e1]), [])
        self.assertEqual(cluster_journal_entries([]), [])

    def test_transitive_clustering(self):
        """A-B overlap + B-C overlap → all three in one cluster via union-find."""
        # Jaccard > 0.3 requires significant overlap. 2/3 shared = 0.5 Jaccard.
        e1 = _make_entry(
            session_id="s1",
            files_modified=["a.py", "shared1.py", "shared2.py"],
        )
        e2 = _make_entry(
            session_id="s2",
            files_modified=["shared1.py", "shared2.py", "shared3.py"],
        )
        e3 = _make_entry(
            session_id="s3",
            files_modified=["shared2.py", "shared3.py", "b.py"],
        )
        clusters = cluster_journal_entries([e1, e2, e3])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), 3)


class TestTemporalWindowClusters(unittest.TestCase):
    """Tests for the temporal fallback clustering."""

    def test_basic_windowing(self):
        entries = [
            _make_entry(session_id=f"s{i}", timestamp=f"2026-03-0{i+1}T10:00:00")
            for i in range(5)
        ]
        clusters = _temporal_window_clusters(entries, window_size=3)
        # 5 entries, window=3, step=2 → windows at [0:3], [2:5]
        self.assertEqual(len(clusters), 2)
        self.assertEqual(len(clusters[0]), 3)
        self.assertEqual(len(clusters[1]), 3)

    def test_two_entries(self):
        entries = [
            _make_entry(session_id="s1", timestamp="2026-03-01T10:00:00"),
            _make_entry(session_id="s2", timestamp="2026-03-02T10:00:00"),
        ]
        clusters = _temporal_window_clusters(entries, window_size=3)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), 2)

    def test_single_entry_returns_empty(self):
        entries = [_make_entry(session_id="s1")]
        self.assertEqual(_temporal_window_clusters(entries), [])

    def test_sorted_by_timestamp(self):
        """Entries should be time-ordered within each window."""
        e1 = _make_entry(session_id="s1", timestamp="2026-03-03T10:00:00")
        e2 = _make_entry(session_id="s2", timestamp="2026-03-01T10:00:00")
        e3 = _make_entry(session_id="s3", timestamp="2026-03-02T10:00:00")
        clusters = _temporal_window_clusters([e1, e2, e3], window_size=3)
        self.assertEqual(len(clusters), 1)
        timestamps = [e.timestamp for e in clusters[0]]
        self.assertEqual(timestamps, sorted(timestamps))

    def test_all_entries_covered(self):
        """Every entry must appear in at least one cluster."""
        entries = [
            _make_entry(session_id=f"s{i}", timestamp=f"2026-03-{i+1:02d}T10:00:00")
            for i in range(7)
        ]
        clusters = _temporal_window_clusters(entries, window_size=3)
        covered = set()
        for cluster in clusters:
            for e in cluster:
                covered.add(e.session_id)
        self.assertEqual(covered, {f"s{i}" for i in range(7)})

    def test_cluster_journal_entries_temporal_fallback(self):
        """cluster_journal_entries uses temporal fallback when no file/keyword overlap."""
        entries = [
            _make_entry(
                session_id=f"s{i}",
                timestamp=f"2026-03-{i+1:02d}T10:00:00",
                focus=f"Unique topic number {i}",
            )
            for i in range(4)
        ]
        clusters = cluster_journal_entries(entries)
        # No file or keyword overlap → temporal fallback
        self.assertGreater(len(clusters), 0)
        # All entries covered
        covered = set()
        for cluster in clusters:
            for e in cluster:
                covered.add(e.session_id)
        self.assertEqual(covered, {f"s{i}" for i in range(4)})


class TestFormatting(unittest.TestCase):
    def test_format_existing_knowledge_empty(self):
        self.assertEqual(_format_existing_knowledge([]), "(none yet)")

    def test_format_existing_knowledge(self):
        node = KnowledgeNode.create(
            content="Always use A100 for training", category="infrastructure"
        )
        text = _format_existing_knowledge([node])
        self.assertIn(node.id, text)
        self.assertIn("infrastructure", text)
        self.assertIn("Always use A100", text)

    def test_format_journal_cluster_sorts_by_timestamp(self):
        e1 = _make_entry(session_id="later-s", timestamp="2026-03-02T00:00:00", focus="Second")
        e2 = _make_entry(session_id="early-s", timestamp="2026-03-01T00:00:00", focus="First")
        text = _format_journal_cluster([e1, e2])
        # Earlier session should appear first in formatted output
        idx_first = text.find("First")
        idx_second = text.find("Second")
        self.assertLess(idx_first, idx_second)

    def test_build_consolidation_prompt(self):
        entry = _make_entry(focus="Working on recall search")
        node = KnowledgeNode.create(content="Use MLX locally", category="tooling")
        prompt = _build_consolidation_prompt([entry], [node])
        self.assertIn("Existing Knowledge", prompt)
        self.assertIn("Use MLX locally", prompt)
        self.assertIn("recall search", prompt)
        self.assertIn("Recent Sessions", prompt)


class TestParseLLMResponse(unittest.TestCase):
    def test_parse_clean_json(self):
        raw = '{"nodes": [{"action": "create", "content": "fact"}]}'
        parsed = _parse_llm_response(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(len(parsed["nodes"]), 1)

    def test_parse_json_with_markdown_fences(self):
        raw = '```json\n{"nodes": []}\n```'
        parsed = _parse_llm_response(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["nodes"], [])

    def test_parse_json_with_surrounding_text(self):
        raw = 'Here is the result:\n{"nodes": [{"action": "create", "content": "fact"}]}\nDone.'
        parsed = _parse_llm_response(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(len(parsed["nodes"]), 1)

    def test_parse_garbage_returns_none(self):
        self.assertIsNone(_parse_llm_response("not json at all"))


class TestApplyConsolidation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.kn_path = Path(self.tmpdir) / "knowledge.jsonl"
        self.cluster = [
            _make_entry(session_id="s1", focus="Session one"),
            _make_entry(session_id="s2", focus="Session two"),
        ]

    def test_create_action(self):
        parsed = {
            "nodes": [{
                "action": "create",
                "content": "Use --iters 500 for cloud training to prevent truncation",
                "category": "convention",
                "confidence": 0.7,
                "tags": ["training", "cloud"],
                "contradiction_note": "",
            }]
        }
        result = _apply_consolidation_result(parsed, [], self.cluster, self.kn_path)
        self.assertEqual(result.nodes_created, 1)
        nodes = read_nodes(self.kn_path)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].content, "Use --iters 500 for cloud training to prevent truncation")
        self.assertEqual(nodes[0].category, "convention")
        self.assertEqual(nodes[0].source_sessions, ["s1", "s2"])

    def test_corroborate_action(self):
        existing = KnowledgeNode.create(
            content="Use A100 for training",
            category="infrastructure",
            source_sessions=["s0"],
            confidence=0.45,
        )
        append_node(existing, self.kn_path)

        parsed = {
            "nodes": [{
                "action": "corroborate",
                "existing_id": existing.id,
                "content": "Use A100 for training",
                "category": "infrastructure",
            }]
        }
        result = _apply_consolidation_result(
            parsed, [existing], self.cluster, self.kn_path,
        )
        self.assertEqual(result.nodes_corroborated, 1)
        self.assertEqual(result.nodes_created, 0)

        # Verify source_sessions updated and confidence bumped
        nodes = read_nodes(self.kn_path)
        self.assertEqual(len(nodes), 1)
        self.assertIn("s0", nodes[0].source_sessions)
        self.assertIn("s1", nodes[0].source_sessions)
        self.assertGreater(nodes[0].confidence, 0.45)

    def test_contradict_action(self):
        old_node = KnowledgeNode.create(
            content="Use MLX for all inference",
            category="tooling",
            source_sessions=["s0"],
        )
        append_node(old_node, self.kn_path)

        parsed = {
            "nodes": [{
                "action": "contradict",
                "existing_id": old_node.id,
                "content": "Use Ollama for inference, MLX for training only",
                "category": "tooling",
                "tags": ["ollama", "mlx"],
                "contradiction_note": "MLX inference too slow for production",
            }]
        }
        result = _apply_consolidation_result(
            parsed, [old_node], self.cluster, self.kn_path,
        )
        self.assertEqual(result.nodes_contradicted, 1)
        self.assertEqual(result.nodes_created, 1)

        # Old node should be contradicted, new node should be active
        all_nodes = read_nodes(self.kn_path)  # All statuses
        active = [n for n in all_nodes if n.status == "active"]
        contradicted = [n for n in all_nodes if n.status == "contradicted"]
        self.assertEqual(len(active), 1)
        self.assertIn("Ollama", active[0].content)
        self.assertEqual(len(contradicted), 1)
        self.assertEqual(contradicted[0].id, old_node.id)

    def test_corroborate_missing_id_becomes_create(self):
        parsed = {
            "nodes": [{
                "action": "corroborate",
                "existing_id": "nonexistent",
                "content": "New fact from bad corroborate",
                "category": "workflow",
            }]
        }
        result = _apply_consolidation_result(parsed, [], self.cluster, self.kn_path)
        self.assertEqual(result.nodes_corroborated, 0)
        self.assertEqual(result.nodes_created, 1)

    def test_empty_content_skipped(self):
        parsed = {"nodes": [{"action": "create", "content": "", "category": "workflow"}]}
        result = _apply_consolidation_result(parsed, [], self.cluster, self.kn_path)
        self.assertEqual(result.nodes_created, 0)

    def test_invalid_nodes_list_ignored(self):
        parsed = {"nodes": "not a list"}
        result = _apply_consolidation_result(parsed, [], self.cluster, self.kn_path)
        self.assertEqual(result.nodes_created, 0)

    def test_multiple_actions_in_one_batch(self):
        existing = KnowledgeNode.create(
            content="Use pytest for testing",
            category="convention",
            source_sessions=["s0"],
        )
        append_node(existing, self.kn_path)

        parsed = {
            "nodes": [
                {
                    "action": "corroborate",
                    "existing_id": existing.id,
                    "content": "Use pytest for testing",
                    "category": "convention",
                },
                {
                    "action": "create",
                    "content": "Use --writer-from flag to freeze writer output across eval runs",
                    "category": "workflow",
                    "confidence": 0.6,
                    "tags": ["eval"],
                },
            ]
        }
        result = _apply_consolidation_result(
            parsed, [existing], self.cluster, self.kn_path,
        )
        self.assertEqual(result.nodes_corroborated, 1)
        self.assertEqual(result.nodes_created, 1)

    def test_content_truncated_at_300_chars(self):
        # Long content (>120 chars) skips specificity check, so plain text is fine
        long_content = "ConnectionPool uses threading.local() for per-thread isolation " + "x" * 500
        parsed = {
            "nodes": [{
                "action": "create",
                "content": long_content,
                "category": "workflow",
            }]
        }
        _apply_consolidation_result(parsed, [], self.cluster, self.kn_path)
        nodes = read_nodes(self.kn_path)
        self.assertLessEqual(len(nodes[0].content), 300)

    def test_create_action_can_disable_temporal_extraction(self):
        parsed = {
            "nodes": [{
                "action": "create",
                "content": "Feature flag rollout starts after PR #42 lands in /src/flags.py",
                "category": "workflow",
                "valid_from": "2026-04-01",
                "valid_until": "2026-04-30",
            }]
        }
        with patch.dict("os.environ", {"SYNAPT_DISABLE_TEMPORAL_EXTRACTION": "1"}):
            result = _apply_consolidation_result(parsed, [], self.cluster, self.kn_path)
        self.assertEqual(result.nodes_created, 1)
        nodes = read_nodes(self.kn_path)
        self.assertEqual(len(nodes), 1)
        self.assertIsNone(nodes[0].valid_until)
        self.assertEqual(nodes[0].valid_from, "2026-03-01T00:00:00")

    def test_create_action_defaults_valid_from_to_earliest_cluster_timestamp(self):
        parsed = {
            "nodes": [{
                "action": "create",
                "content": "Release flag flipped after PR #42 landed in /src/flags.py",
                "category": "workflow",
            }]
        }
        result = _apply_consolidation_result(parsed, [], self.cluster, self.kn_path)
        self.assertEqual(result.nodes_created, 1)
        nodes = read_nodes(self.kn_path)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].valid_from, "2026-03-01T00:00:00")


class TestIsGenericNode(unittest.TestCase):
    """Test the generic advice quality filter."""

    def test_rejects_docker(self):
        self.assertTrue(_is_generic_node("Use Docker for containerization"))
        self.assertTrue(_is_generic_node("Always use Docker"))
        self.assertTrue(_is_generic_node("Prefer Docker"))

    def test_rejects_naming_convention(self):
        self.assertTrue(_is_generic_node("Use a consistent naming convention"))
        self.assertTrue(_is_generic_node("Use consistent naming for variables"))

    def test_rejects_generic_tests(self):
        self.assertTrue(_is_generic_node("Always write tests"))
        self.assertTrue(_is_generic_node("Use unit tests"))

    def test_rejects_generic_gpu(self):
        self.assertTrue(_is_generic_node("Use GPU for training"))
        self.assertTrue(_is_generic_node("Use GPU with at least 8GB"))

    def test_accepts_specific_gpu(self):
        self.assertFalse(_is_generic_node("Use A100 for training 8B models"))
        self.assertFalse(_is_generic_node("Use A10G for eval, A100 for training"))

    def test_accepts_project_specific(self):
        self.assertFalse(_is_generic_node("Train on Alfred eval set, test on Batman"))
        self.assertFalse(_is_generic_node("Use --iters 500 for cloud training"))
        self.assertFalse(_is_generic_node("Each language gets its own adapter pair"))
        self.assertFalse(_is_generic_node("Run verify_quality_curve.py before training"))
        self.assertFalse(_is_generic_node("Package renamed from synapse to synapt"))

    def test_rejects_best_practices(self):
        self.assertTrue(_is_generic_node("Follow best practices for code"))
        self.assertTrue(_is_generic_node("Use coding standards"))

    def test_rejects_documentation_advice(self):
        self.assertTrue(_is_generic_node("Always document your code"))
        self.assertTrue(_is_generic_node("Comment the functions"))

    def test_rejects_clean_code(self):
        self.assertTrue(_is_generic_node("Keep code clean and simple"))
        self.assertTrue(_is_generic_node("Write functions small"))


    def test_rejects_tool_tautology(self):
        """Tool used for its primary purpose is generic."""
        self.assertTrue(_is_generic_node("Use gradlew to build the project"))
        self.assertTrue(_is_generic_node("Use npm to install dependencies"))
        self.assertTrue(_is_generic_node("Use pip to install packages"))
        self.assertTrue(_is_generic_node("Use pytest for testing"))
        self.assertTrue(_is_generic_node("Use black for formatting"))
        self.assertTrue(_is_generic_node("Use eslint to check code"))

    def test_rejects_generic_config(self):
        """Generic config file knowledge."""
        self.assertTrue(_is_generic_node("Configure settings.gradle for the build"))
        self.assertTrue(_is_generic_node("Set up pyproject.toml for the project"))

    def test_rejects_generic_workflow(self):
        """Generic workflow advice."""
        self.assertTrue(_is_generic_node("Review code before merging"))
        self.assertTrue(_is_generic_node("Handle errors gracefully"))
        self.assertTrue(_is_generic_node("Keep dependencies up to date"))

    def test_tool_tautology_accepts_specific(self):
        """Tool-tautology patterns should NOT reject content with specificity signals."""
        self.assertFalse(_is_generic_node("Use pip to install synapt[dev] from local path"))
        self.assertFalse(_is_generic_node("Use pytest to test with -x flag for fail-fast"))
        self.assertFalse(_is_generic_node("Use npm to install @types/react@18.2"))
        self.assertFalse(_is_generic_node("Use gradle to build the :app:debug variant with --stacktrace"))


class TestLacksSpecificity(unittest.TestCase):
    """Test the specificity signal detection for filtering generic knowledge."""

    def test_generic_tool_knowledge(self):
        """Tool knowledge without project-specific signals is generic."""
        self.assertTrue(_lacks_specificity("Use Gradle for building Android apps"))
        self.assertTrue(_lacks_specificity("Store secrets in environment variables"))
        self.assertTrue(_lacks_specificity("Run tests before deploying to production"))

    def test_specific_with_path(self):
        """Content with file paths is specific."""
        self.assertFalse(_lacks_specificity("Store config in /src/config/models.json"))
        self.assertFalse(_lacks_specificity("Run ./scripts/verify_quality_curve.py"))

    def test_specific_with_version(self):
        """Content with version numbers is specific."""
        self.assertFalse(_lacks_specificity("Pin croniter to v1.3.8 in pyproject.toml"))
        self.assertFalse(_lacks_specificity("Upgraded from 3.2.1 to 4.0.0"))

    def test_specific_with_cli_flag(self):
        """Content with CLI flags is specific."""
        self.assertFalse(_lacks_specificity("Use --full-pipeline for LOCOMO eval"))
        self.assertFalse(_lacks_specificity("The --writer-from flag freezes output"))

    def test_specific_with_camelcase(self):
        """Content with CamelCase identifiers is specific."""
        self.assertFalse(_lacks_specificity("ConnectionPool uses threading.local()"))
        self.assertFalse(_lacks_specificity("KnowledgeNode stores source_sessions"))

    def test_specific_with_snake_case(self):
        """Content with multi-part snake_case identifiers is specific."""
        self.assertFalse(_lacks_specificity("The source_turns field links to transcripts"))

    def test_specific_with_session_ref(self):
        """Content with session/PR/issue references is specific."""
        self.assertFalse(_lacks_specificity("Decided in session 7 to use WAL mode"))
        self.assertFalse(_lacks_specificity("Fixed in PR #25 with rstrip fix"))

    def test_specific_with_date(self):
        """Content with dates is specific."""
        self.assertFalse(_lacks_specificity("Merge freeze begins 2026-03-05"))

    def test_long_content_passes(self):
        """Content > 120 chars always passes (assumed to have context)."""
        long = "Use Gradle for building" + " and more detail" * 8  # > 120 chars
        self.assertFalse(_lacks_specificity(long))


    def test_code_generic_tool_output_rejected(self):
        """Code-specific tool output noise is rejected when content_type='code'."""
        self.assertTrue(_lacks_specificity("build succeeded", content_type="code"))
        self.assertTrue(_lacks_specificity("all tests passed", content_type="code"))
        self.assertTrue(_lacks_specificity("file saved successfully", content_type="code"))
        self.assertTrue(_lacks_specificity("linting passed", content_type="code"))
        self.assertTrue(_lacks_specificity("no errors found", content_type="code"))

    def test_code_generic_not_rejected_without_content_type(self):
        """Code-specific patterns don't fire without content_type='code'."""
        # These are short + lack specificity signals, so they still fail the
        # general filter — but the code-specific patterns shouldn't be the reason.
        # Test with content_type=None (general filter still catches them).
        self.assertTrue(_lacks_specificity("Build succeeded"))
        # With content_type="personal", code patterns don't apply
        self.assertTrue(_lacks_specificity("Build succeeded", content_type="personal"))

    def test_code_specific_tool_output_preserved(self):
        """Tool output with project-specific signals survives even for code."""
        self.assertFalse(_lacks_specificity(
            "Build succeeded for src/synapt/recall/core.py", content_type="code"
        ))
        self.assertFalse(_lacks_specificity(
            "Deployed v0.7.8 to PyPI", content_type="code"
        ))

    def test_lower_code_threshold(self):
        """Code content uses stricter 80-char threshold via adaptive_params."""
        from synapt.recall.content_profile import ContentProfile, adaptive_params
        code_profile = ContentProfile(total_chunks=10, file_refs=30, tool_uses=20)
        self.assertTrue(code_profile.is_code)
        ap = adaptive_params(code_profile)
        self.assertEqual(ap.specificity_threshold, 80)

    def test_personal_threshold_disabled(self):
        """Personal content has effectively disabled specificity filter."""
        from synapt.recall.content_profile import ContentProfile, adaptive_params
        personal_profile = ContentProfile(total_chunks=10, personal_refs=20, name_addresses=5)
        self.assertTrue(personal_profile.is_personal)
        ap = adaptive_params(personal_profile)
        self.assertEqual(ap.specificity_threshold, 10000)


class TestGenericFilterInApply(unittest.TestCase):
    """Test that generic nodes are rejected during application."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.kn_path = Path(self.tmpdir) / "knowledge.jsonl"
        self.cluster = [
            _make_entry(session_id="s1", focus="Session one"),
            _make_entry(session_id="s2", focus="Session two"),
        ]

    def test_generic_create_rejected(self):
        parsed = {
            "nodes": [{
                "action": "create",
                "content": "Always use Docker for containerization",
                "category": "tooling",
                "confidence": 0.7,
                "tags": ["docker"],
            }]
        }
        result = _apply_consolidation_result(parsed, [], self.cluster, self.kn_path)
        self.assertEqual(result.nodes_created, 0)
        nodes = read_nodes(self.kn_path)
        self.assertEqual(len(nodes), 0)

    def test_low_specificity_create_rejected(self):
        """Short content without specificity signals should be rejected."""
        parsed = {
            "nodes": [{
                "action": "create",
                "content": "Store secrets in environment variables for safety",
                "category": "convention",
                "confidence": 0.7,
                "tags": [],
            }]
        }
        result = _apply_consolidation_result(parsed, [], self.cluster, self.kn_path)
        self.assertEqual(result.nodes_created, 0)
        nodes = read_nodes(self.kn_path)
        self.assertEqual(len(nodes), 0)

    def test_garbled_source_turns_rejected(self):
        """Content that is just source turn references should be rejected."""
        parsed = {
            "nodes": [{
                "action": "create",
                "content": "Source turns: s011c00:5, s013c00:12",
                "category": "fact",
                "confidence": 0.6,
                "tags": [],
            }]
        }
        result = _apply_consolidation_result(parsed, [], self.cluster, self.kn_path)
        self.assertEqual(result.nodes_created, 0)

    def test_garbled_inline_metadata_rejected(self):
        """Content with inline LLM metadata (existing_id, etc.) should be rejected."""
        parsed = {
            "nodes": [{
                "action": "create",
                "content": '"Caroline adopted a dog" (fact, corroborates "Caroline got a pet") (existing_id: 05e27ef8ff8e)',
                "category": "fact",
                "confidence": 0.6,
                "tags": [],
            }]
        }
        result = _apply_consolidation_result(parsed, [], self.cluster, self.kn_path)
        self.assertEqual(result.nodes_created, 0)

    def test_garbled_verified_annotation_rejected(self):
        """Content with '(fact) - verified' annotation should be rejected."""
        parsed = {
            "nodes": [{
                "action": "create",
                "content": "Caroline adopted a child in 2025 (fact) - verified across 2+ sessions",
                "category": "fact",
                "confidence": 0.6,
                "tags": [],
            }]
        }
        result = _apply_consolidation_result(parsed, [], self.cluster, self.kn_path)
        self.assertEqual(result.nodes_created, 0)

    def test_garbled_detection_unit(self):
        """Unit tests for _is_garbled_content patterns."""
        assert _is_garbled_content("Source turns: s011c00:5, s013c00:12")
        assert _is_garbled_content('existing_id: 05e27ef8ff8e')
        assert _is_garbled_content('"X did Y" (fact, corroborates "X did Z")')
        assert _is_garbled_content("X adopted a dog (fact) - verified across sessions")
        # Non-garbled content should pass
        assert not _is_garbled_content("Caroline adopted a rescue dog named Rex")
        assert not _is_garbled_content("Melanie prefers hiking over camping")
        assert not _is_garbled_content("API listens on port 5432")

    def test_specific_create_accepted(self):
        parsed = {
            "nodes": [{
                "action": "create",
                "content": "Use A100 with --batch-size 4 for 8B model training",
                "category": "infrastructure",
                "confidence": 0.7,
                "tags": ["gpu"],
            }]
        }
        result = _apply_consolidation_result(parsed, [], self.cluster, self.kn_path)
        self.assertEqual(result.nodes_created, 1)

    def test_auto_corroborate_near_duplicate(self):
        """Create with content similar to existing node should auto-corroborate."""
        existing = KnowledgeNode.create(
            content="Use --phase-filter config options for phase filtering and custom prompts for L3 repair loop",
            category="architecture",
            source_sessions=["s0"],
        )
        append_node(existing, self.kn_path)
        parsed = {
            "nodes": [{
                "action": "create",
                "content": "Use --phase-filter config options for phase filtering (e.g., runtime errors) and custom prompts for L3 repair",
                "category": "architecture",
                "confidence": 0.7,
                "tags": ["repair"],
            }]
        }
        result = _apply_consolidation_result(
            parsed, [existing], self.cluster, self.kn_path,
        )
        self.assertEqual(result.nodes_created, 0)
        self.assertEqual(result.nodes_corroborated, 1)
        # Original node should still be the only one
        nodes = read_nodes(self.kn_path)
        self.assertEqual(len(nodes), 1)

    def test_no_auto_corroborate_different_content(self):
        """Create with dissimilar content should not auto-corroborate."""
        existing = KnowledgeNode.create(
            content="Use A100 with --batch-size 4 for 8B model training — A10G OOMs",
            category="infrastructure",
            source_sessions=["s0"],
        )
        append_node(existing, self.kn_path)
        parsed = {
            "nodes": [{
                "action": "create",
                "content": "Package renamed from synapse to synapt (see PR #12)",
                "category": "decision",
                "confidence": 0.8,
                "tags": ["naming"],
            }]
        }
        result = _apply_consolidation_result(
            parsed, [existing], self.cluster, self.kn_path,
        )
        self.assertEqual(result.nodes_created, 1)
        self.assertEqual(result.nodes_corroborated, 0)

    def test_auto_corroborate_at_exact_boundary(self):
        """Jaccard of exactly 0.5 should trigger auto-corroborate (>= threshold)."""
        # keywords("alpha bravo") = {"alpha","bravo"}, |inter|=2
        # keywords("alpha bravo --charlie delta") = {"alpha","bravo","charlie","delta"}, |union|=4
        # jaccard = 2/4 = 0.5 ✓
        existing = KnowledgeNode.create(
            content="alpha bravo",
            category="convention",
            source_sessions=["s0"],
        )
        append_node(existing, self.kn_path)
        parsed = {
            "nodes": [{
                "action": "create",
                "content": "alpha bravo --charlie delta",
                "category": "convention",
                "confidence": 0.7,
                "tags": [],
            }]
        }
        result = _apply_consolidation_result(
            parsed, [existing], self.cluster, self.kn_path,
        )
        self.assertEqual(result.nodes_corroborated, 1)
        self.assertEqual(result.nodes_created, 0)

    def test_intra_batch_dedup(self):
        """Two creates in the same LLM response with similar content: second should auto-corroborate against first."""
        parsed = {
            "nodes": [
                {
                    "action": "create",
                    "content": "Use Modal with --gpu a10g for cloud GPU training runs",
                    "category": "infrastructure",
                    "confidence": 0.7,
                    "tags": ["modal", "training"],
                },
                {
                    "action": "create",
                    "content": "Use Modal with --gpu a10g for cloud GPU training and evaluation",
                    "category": "infrastructure",
                    "confidence": 0.6,
                    "tags": ["modal", "eval"],
                },
            ]
        }
        result = _apply_consolidation_result(
            parsed, [], self.cluster, self.kn_path,
        )
        # First create succeeds, second auto-corroborates against it
        self.assertEqual(result.nodes_created, 1)
        self.assertEqual(result.nodes_corroborated, 1)
        nodes = read_nodes(self.kn_path)
        self.assertEqual(len(nodes), 1)

    def test_embedding_auto_corroborate_semantic_duplicate(self):
        """Semantic duplicate (different wording, same meaning) should auto-corroborate via embeddings."""
        import synapt.recall.consolidate as mod

        # Mock the embedding dedup to simulate high cosine similarity
        # for semantically similar but keyword-different content.
        original_fn = mod._inline_embedding_dedup
        existing = KnowledgeNode.create(
            content="Kotlin Multiplatform projects are linked to Xcode via build_phases for iOS builds",
            category="architecture",
            source_sessions=["s0"],
        )
        append_node(existing, self.kn_path)

        def mock_emb_dedup(candidate, existing_nodes, threshold=0.80):
            # Simulate: "KMP frameworks linked to Xcode" is semantically
            # similar to existing content (cosine=0.88) but keyword-different
            # enough that Jaccard < 0.5.
            if "KMP" in candidate and existing_nodes:
                return (existing_nodes[0], 0.88)
            return (None, 0.0)

        mod._inline_embedding_dedup = mock_emb_dedup
        try:
            parsed = {
                "nodes": [{
                    "action": "create",
                    "content": "KMP frameworks linked to Xcode for native iOS integration via CocoaPods",
                    "category": "architecture",
                    "confidence": 0.7,
                    "tags": [],
                }]
            }
            result = _apply_consolidation_result(
                parsed, [existing], self.cluster, self.kn_path,
            )
            self.assertEqual(result.nodes_corroborated, 1)
            self.assertEqual(result.nodes_created, 0)
        finally:
            mod._inline_embedding_dedup = original_fn

    def test_embedding_below_threshold_still_creates(self):
        """Low cosine similarity should NOT auto-corroborate; node is created normally."""
        import synapt.recall.consolidate as mod

        existing = KnowledgeNode.create(
            content="Kotlin Multiplatform projects are linked to Xcode via build_phases for iOS builds",
            category="architecture",
            source_sessions=["s0"],
        )
        append_node(existing, self.kn_path)

        original_fn = mod._inline_embedding_dedup

        def mock_emb_dedup(candidate, existing_nodes, threshold=0.80):
            # Simulate: cosine similarity below threshold
            if existing_nodes:
                return (None, 0.60)
            return (None, 0.0)

        mod._inline_embedding_dedup = mock_emb_dedup
        try:
            parsed = {
                "nodes": [{
                    "action": "create",
                    "content": "Gradle builds use the Kotlin DSL with settings_gradle for configuration",
                    "category": "tooling",
                    "confidence": 0.7,
                    "tags": [],
                }]
            }
            result = _apply_consolidation_result(
                parsed, [existing], self.cluster, self.kn_path,
            )
            self.assertEqual(result.nodes_created, 1)
            self.assertEqual(result.nodes_corroborated, 0)
        finally:
            mod._inline_embedding_dedup = original_fn

    def test_generic_contradict_rejected(self):
        """Contradict with generic replacement content should be rejected."""
        existing = KnowledgeNode.create(
            content="Use Ollama for inference",
            category="tooling",
            source_sessions=["s0"],
        )
        append_node(existing, self.kn_path)
        parsed = {
            "nodes": [{
                "action": "contradict",
                "existing_id": existing.id,
                "content": "Always use Docker for containerization",
                "category": "tooling",
                "contradiction_note": "Switched to Docker",
            }]
        }
        result = _apply_consolidation_result(
            parsed, [existing], self.cluster, self.kn_path,
        )
        self.assertEqual(result.nodes_created, 0)
        self.assertEqual(result.nodes_contradicted, 0)
        # Original node should remain active (not contradicted)
        nodes = read_nodes(self.kn_path)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].status, "active")

    def test_generic_filter_does_not_block_corroborate(self):
        """Corroborate actions should not be filtered — the node already passed."""
        existing = KnowledgeNode.create(
            content="Use pytest for testing",
            category="convention",
            source_sessions=["s0"],
        )
        append_node(existing, self.kn_path)
        parsed = {
            "nodes": [{
                "action": "corroborate",
                "existing_id": existing.id,
                "content": "Use pytest for testing",
                "category": "convention",
            }]
        }
        result = _apply_consolidation_result(
            parsed, [existing], self.cluster, self.kn_path,
        )
        self.assertEqual(result.nodes_corroborated, 1)


class TestProjectContext(unittest.TestCase):
    """Test project context extraction for prompt grounding."""

    def test_get_project_context_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _get_project_context(Path(tmp))
            self.assertIn(Path(tmp).name, ctx)

    def test_get_project_context_with_claude_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            claude_md = Path(tmp) / "CLAUDE.md"
            claude_md.write_text("# My Project\n\nThis is a multi-model orchestrator.\n")
            ctx = _get_project_context(Path(tmp))
            self.assertIn("multi-model orchestrator", ctx)

    def test_prompt_includes_project_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = _make_entry(focus="Working on recall search")
            prompt = _build_consolidation_prompt([entry], [], Path(tmp))
            self.assertIn("Project Context", prompt)
            self.assertIn(Path(tmp).name, prompt)

    def test_prompt_includes_few_shot_examples(self):
        entry = _make_entry(focus="Working on recall search")
        prompt = _build_consolidation_prompt([entry], [])
        self.assertIn("GOOD knowledge nodes", prompt)
        self.assertIn("BAD knowledge nodes", prompt)
        self.assertIn("NEVER produce these", prompt)

    def test_prompt_includes_strong_rules(self):
        entry = _make_entry(focus="Working on recall search")
        prompt = _build_consolidation_prompt([entry], [])
        self.assertIn("Do NOT extract generic advice", prompt)
        self.assertIn("Empty is better than generic", prompt)


class TestBuildFewShotExamples(unittest.TestCase):
    """Test dynamic few-shot example selection."""

    def test_fallback_to_defaults_when_no_nodes(self):
        result = _build_few_shot_examples([])
        for default in _DEFAULT_GOOD_EXAMPLES:
            self.assertIn(default, result)

    def test_uses_existing_nodes(self):
        nodes = [
            KnowledgeNode.create(
                content="Use Modal for cloud training",
                category="infrastructure",
                confidence=0.8,
            ),
        ]
        result = _build_few_shot_examples(nodes)
        self.assertIn("Use Modal for cloud training", result)
        self.assertIn("infrastructure", result)
        # Should NOT contain hardcoded defaults
        for default in _DEFAULT_GOOD_EXAMPLES:
            self.assertNotIn(default, result)

    def test_category_diversity(self):
        """Should pick highest-confidence node per category, not first-seen."""
        nodes = [
            KnowledgeNode.create(content="Fact B infra low", category="infrastructure", confidence=0.5),
            KnowledgeNode.create(content="Fact A infra high", category="infrastructure", confidence=0.9),
            KnowledgeNode.create(content="Fact C workflow", category="workflow", confidence=0.7),
        ]
        result = _build_few_shot_examples(nodes, max_examples=4)
        # Should have Fact A (highest confidence infra) and Fact C (workflow)
        self.assertIn("Fact A infra high", result)
        self.assertIn("Fact C workflow", result)
        # Fact B should be excluded (same category as Fact A, lower confidence)
        self.assertNotIn("Fact B infra low", result)

    def test_prompt_uses_dynamic_examples(self):
        """Full integration: prompt should include dynamic examples from nodes."""
        node = KnowledgeNode.create(
            content="Always run pytest before merging PRs",
            category="workflow",
            confidence=0.8,
        )
        entry = _make_entry(focus="Working on recall search")
        prompt = _build_consolidation_prompt([entry], [node])
        self.assertIn("Always run pytest before merging PRs", prompt)
        self.assertIn("GOOD knowledge nodes", prompt)


class TestSplitLargeCluster(unittest.TestCase):
    """Test mega-cluster splitting into time-ordered sub-clusters."""

    def test_small_cluster_unchanged(self):
        """Clusters <= max_size should be returned as-is."""
        entries = [_make_entry(session_id=f"s{i}") for i in range(4)]
        result = _split_large_cluster(entries, max_size=6)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], entries)

    def test_exact_max_size_unchanged(self):
        entries = [_make_entry(session_id=f"s{i}") for i in range(6)]
        result = _split_large_cluster(entries, max_size=6)
        self.assertEqual(len(result), 1)

    def test_splits_large_cluster(self):
        """Cluster of 12 with max_size=6 should produce 3 sub-clusters."""
        entries = [
            _make_entry(session_id=f"s{i:02d}", timestamp=f"2026-03-{i+1:02d}T00:00:00")
            for i in range(12)
        ]
        result = _split_large_cluster(entries, max_size=6)
        # step = 5 (max_size - 1), so windows start at 0, 5, 10
        # window 0: entries 0-5 (6 entries)
        # window 5: entries 5-10 (6 entries)
        # window 10: entries 10-11 (2 entries)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 6)
        self.assertEqual(len(result[1]), 6)
        self.assertEqual(len(result[2]), 2)

    def test_windows_overlap_by_one(self):
        """Adjacent windows should share exactly 1 entry for context continuity."""
        entries = [
            _make_entry(session_id=f"s{i:02d}", timestamp=f"2026-03-{i+1:02d}T00:00:00")
            for i in range(12)
        ]
        result = _split_large_cluster(entries, max_size=6)
        # Last entry of window 0 should be first entry of window 1
        ids_0 = [e.session_id for e in result[0]]
        ids_1 = [e.session_id for e in result[1]]
        overlap = set(ids_0) & set(ids_1)
        self.assertEqual(len(overlap), 1)

    def test_time_ordering(self):
        """Entries should be sorted by timestamp regardless of input order."""
        entries = [
            _make_entry(session_id="late", timestamp="2026-03-10T00:00:00"),
            _make_entry(session_id="early", timestamp="2026-03-01T00:00:00"),
            _make_entry(session_id="mid", timestamp="2026-03-05T00:00:00"),
        ]
        result = _split_large_cluster(entries, max_size=6)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0].session_id, "early")
        self.assertEqual(result[0][1].session_id, "mid")
        self.assertEqual(result[0][2].session_id, "late")

    def test_all_entries_covered(self):
        """Every entry must appear in at least one sub-cluster."""
        entries = [
            _make_entry(session_id=f"s{i:02d}", timestamp=f"2026-03-{i+1:02d}T00:00:00")
            for i in range(20)
        ]
        result = _split_large_cluster(entries, max_size=6)
        all_ids = set()
        for sub in result:
            for e in sub:
                all_ids.add(e.session_id)
        expected_ids = {f"s{i:02d}" for i in range(20)}
        self.assertEqual(all_ids, expected_ids)

    def test_minimum_window_size(self):
        """Trailing windows with < 2 entries should be dropped."""
        # 7 entries, max_size=6 → step=5
        # window 0: entries 0-5 (6 entries)
        # window 5: entries 5-6 (2 entries) — kept (>= 2)
        entries = [
            _make_entry(session_id=f"s{i}", timestamp=f"2026-03-{i+1:02d}T00:00:00")
            for i in range(7)
        ]
        result = _split_large_cluster(entries, max_size=6)
        self.assertEqual(len(result), 2)
        for sub in result:
            self.assertGreaterEqual(len(sub), 2)


class TestDedupDecisionLogging(unittest.TestCase):
    """Test pairwise decision logging for future dedup adapter training."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.kn_path = Path(self.tmp) / "knowledge.jsonl"
        self.decision_path = Path(self.tmp) / "dedup_decisions.jsonl"

    def _read_decisions(self) -> list[dict]:
        import json
        if not self.decision_path.exists():
            return []
        lines = []
        for line in self.decision_path.open():
            line = line.strip()
            if line:
                lines.append(json.loads(line))
        return lines

    def test_corroborate_logs_decision(self):
        existing = KnowledgeNode.create(
            content="Use A100 for training", category="infrastructure",
            source_sessions=["s0"],
        )
        append_node(existing, self.kn_path)
        parsed = {"nodes": [{
            "action": "corroborate",
            "existing_id": existing.id,
            "content": "A100 is required for training",
            "category": "infrastructure",
        }]}
        cluster = [_make_entry(session_id="s1")]
        _apply_consolidation_result(
            parsed, [existing], cluster, self.kn_path,
            decision_log_path=self.decision_path,
        )
        decisions = self._read_decisions()
        self.assertEqual(len(decisions), 1)
        d = decisions[0]
        self.assertEqual(d["action"], "corroborate")
        self.assertEqual(d["source"], "llm")
        self.assertEqual(d["existing_id"], existing.id)
        self.assertEqual(d["existing_content"], "Use A100 for training")
        self.assertIn("timestamp", d)

    def test_contradict_logs_decision(self):
        old_node = KnowledgeNode.create(
            content="Use MLX for all inference", category="tooling",
            source_sessions=["s0"],
        )
        append_node(old_node, self.kn_path)
        parsed = {"nodes": [{
            "action": "contradict",
            "existing_id": old_node.id,
            "content": "Use Ollama for local inference instead of MLX",
            "category": "tooling",
            "contradiction_note": "MLX is no longer maintained",
        }]}
        cluster = [_make_entry(session_id="s1")]
        _apply_consolidation_result(
            parsed, [old_node], cluster, self.kn_path,
            decision_log_path=self.decision_path,
        )
        decisions = self._read_decisions()
        self.assertEqual(len(decisions), 1)
        d = decisions[0]
        self.assertEqual(d["action"], "contradict")
        self.assertEqual(d["source"], "llm")
        self.assertEqual(d["existing_id"], old_node.id)
        self.assertIn("contradiction_note", d)

    def test_auto_corroborate_logs_decision(self):
        """Jaccard >= 0.5 auto-corroborate should log with similarity score."""
        existing = KnowledgeNode.create(
            content="alpha bravo charlie delta echo --verbose",
            category="convention", source_sessions=["s0"],
        )
        append_node(existing, self.kn_path)
        # Content shares enough keywords to trigger Jaccard >= 0.5
        parsed = {"nodes": [{
            "action": "create",
            "content": "alpha bravo charlie delta foxtrot --verbose",
            "category": "convention",
        }]}
        cluster = [_make_entry(session_id="s1")]
        _apply_consolidation_result(
            parsed, [existing], cluster, self.kn_path,
            decision_log_path=self.decision_path,
        )
        decisions = self._read_decisions()
        self.assertEqual(len(decisions), 1)
        d = decisions[0]
        self.assertEqual(d["action"], "auto-corroborate")
        self.assertEqual(d["source"], "auto-jaccard")
        self.assertGreaterEqual(d["similarity_score"], 0.5)
        self.assertEqual(d["existing_id"], existing.id)

    def test_create_logs_negative_pairs(self):
        """Create with existing nodes logs top-3 negative pairs by Jaccard."""
        nodes = [
            KnowledgeNode.create(
                content=f"unique-word-{i} common-term shared-idea",
                category="tooling", source_sessions=["s0"],
            )
            for i in range(4)
        ]
        for n in nodes:
            append_node(n, self.kn_path)
        # Completely different content — all below 0.5 threshold
        parsed = {"nodes": [{
            "action": "create",
            "content": "completely different zebra xylophone --quantum-mode",
            "category": "workflow",
        }]}
        cluster = [_make_entry(session_id="s1")]
        _apply_consolidation_result(
            parsed, list(nodes), cluster, self.kn_path,
            decision_log_path=self.decision_path,
        )
        decisions = self._read_decisions()
        self.assertEqual(len(decisions), 1)
        d = decisions[0]
        self.assertEqual(d["action"], "create")
        # Negative pairs may be empty if no Jaccard > 0 matches
        # (the words are totally different)
        # But if any share keywords, there would be pairs

    def test_create_no_existing_nodes(self):
        """Create with no existing nodes — no negative_pairs field."""
        parsed = {"nodes": [{
            "action": "create",
            "content": "Brand new knowledge about testing patterns with --coverage",
            "category": "workflow",
        }]}
        cluster = [_make_entry(session_id="s1")]
        _apply_consolidation_result(
            parsed, [], cluster, self.kn_path,
            decision_log_path=self.decision_path,
        )
        decisions = self._read_decisions()
        self.assertEqual(len(decisions), 1)
        d = decisions[0]
        self.assertEqual(d["action"], "create")
        self.assertNotIn("negative_pairs", d)

    def test_decision_log_valid_jsonl(self):
        """All decision log entries should be valid JSON with required fields."""
        import json
        existing = KnowledgeNode.create(
            content="Use A100 for training", category="infrastructure",
            source_sessions=["s0"],
        )
        append_node(existing, self.kn_path)
        # Multiple actions in one batch
        parsed = {"nodes": [
            {"action": "corroborate", "existing_id": existing.id,
             "content": "A100 needed", "category": "infrastructure"},
            {"action": "create", "content": "New pattern about linting tools with --fix flag",
             "category": "workflow"},
        ]}
        cluster = [_make_entry(session_id="s1")]
        _apply_consolidation_result(
            parsed, [existing], cluster, self.kn_path,
            decision_log_path=self.decision_path,
        )
        decisions = self._read_decisions()
        self.assertEqual(len(decisions), 2)
        required_fields = {"timestamp", "action", "candidate_content",
                           "candidate_category", "session_ids", "source"}
        for d in decisions:
            self.assertTrue(required_fields.issubset(d.keys()),
                            f"Missing fields: {required_fields - d.keys()}")

    def test_no_logging_when_path_is_none(self):
        """No decision log file should be created when path is None."""
        parsed = {"nodes": [{
            "action": "create",
            "content": "Some new knowledge about patterns",
            "category": "workflow",
        }]}
        cluster = [_make_entry(session_id="s1")]
        _apply_consolidation_result(
            parsed, [], cluster, self.kn_path,
            # decision_log_path defaults to None
        )
        self.assertFalse(self.decision_path.exists())

    def test_log_dedup_decision_direct(self):
        """Direct call to _log_dedup_decision produces correct JSONL."""
        _log_dedup_decision(
            self.decision_path,
            action="auto-corroborate",
            candidate_content="Test node content",
            candidate_category="tooling",
            existing_id="abc123",
            existing_content="Existing node",
            similarity_score=0.73456789,
            source="auto-jaccard",
            session_ids=["s1", "s2"],
        )
        decisions = self._read_decisions()
        self.assertEqual(len(decisions), 1)
        d = decisions[0]
        self.assertEqual(d["similarity_score"], 0.7346)  # Rounded to 4dp
        self.assertEqual(d["session_ids"], ["s1", "s2"])
        self.assertEqual(d["existing_id"], "abc123")


class TestResponseCache(unittest.TestCase):
    """Test cluster-level LLM response caching."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cache_path = Path(self.tmp) / "consolidation_cache.jsonl"

    def test_cache_key_deterministic(self):
        """Same entries in any order produce the same key."""
        e1 = _make_entry(session_id="s1", timestamp="2026-03-01T00:00:00")
        e2 = _make_entry(session_id="s2", timestamp="2026-03-02T00:00:00")
        key_a = _cluster_cache_key([e1, e2])
        key_b = _cluster_cache_key([e2, e1])
        self.assertEqual(key_a, key_b)

    def test_cache_key_different_for_different_clusters(self):
        e1 = _make_entry(session_id="s1")
        e2 = _make_entry(session_id="s2")
        e3 = _make_entry(session_id="s3")
        self.assertNotEqual(
            _cluster_cache_key([e1, e2]),
            _cluster_cache_key([e1, e3]),
        )

    def test_save_and_load(self):
        _save_cached_response(self.cache_path, "abc123", '{"nodes": []}', "the prompt")
        cache = _load_response_cache(self.cache_path)
        self.assertEqual(cache["abc123"]["response"], '{"nodes": []}')
        self.assertEqual(cache["abc123"]["prompt"], "the prompt")

    def test_load_empty_cache(self):
        cache = _load_response_cache(self.cache_path)
        self.assertEqual(cache, {})

    def test_multiple_entries(self):
        _save_cached_response(self.cache_path, "k1", '{"nodes": [{"action": "create"}]}', "p1")
        _save_cached_response(self.cache_path, "k2", '{"nodes": []}', "p2")
        cache = _load_response_cache(self.cache_path)
        self.assertEqual(len(cache), 2)
        self.assertIn("k1", cache)
        self.assertIn("k2", cache)

    def test_corrupt_lines_skipped(self):
        """Malformed lines in cache file are silently skipped."""
        self.cache_path.write_text('not json\n{"key": "k1", "response": "ok"}\n')
        cache = _load_response_cache(self.cache_path)
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache["k1"]["response"], "ok")

    def test_backwards_compatible_without_prompt(self):
        """Cache entries without prompt field still load correctly."""
        import json as _json
        self.cache_path.write_text(
            _json.dumps({"key": "old", "response": '{"nodes": []}'}) + "\n"
        )
        cache = _load_response_cache(self.cache_path)
        self.assertEqual(cache["old"]["response"], '{"nodes": []}')
        self.assertEqual(cache["old"]["prompt"], "")


class TestRelevanceFilteredKnowledge(unittest.TestCase):
    """Tests for cluster-aware relevance filtering in _format_existing_knowledge."""

    def _make_node(self, content, category="tooling", confidence=0.5):
        return KnowledgeNode.create(
            content=content, category=category, confidence=confidence,
        )

    def test_filters_by_cluster_relevance(self):
        """Relevant nodes appear before irrelevant ones."""
        relevant = self._make_node("Use MLX adapter for swift repair")
        irrelevant = self._make_node("Docker compose for Postgres setup")
        cluster = [_make_entry(focus="Training swift repair adapter")]
        text = _format_existing_knowledge(
            [irrelevant, relevant], cluster=cluster, max_relevant=8,
        )
        # Both appear (only 2 nodes, both fit in max_relevant=8)
        self.assertIn("swift repair", text)
        self.assertIn("Docker compose", text)
        # Relevant node appears first
        self.assertLess(text.index("swift repair"), text.index("Docker compose"))

    def test_cluster_none_backward_compat(self):
        """Without a cluster, all nodes are shown (original behaviour)."""
        nodes = [self._make_node(f"Node {i}") for i in range(3)]
        text_no_cluster = _format_existing_knowledge(nodes)
        text_none = _format_existing_knowledge(nodes, cluster=None)
        self.assertEqual(text_no_cluster, text_none)
        for i in range(3):
            self.assertIn(f"Node {i}", text_no_cluster)

    def test_shows_omitted_count(self):
        """When nodes exceed max_relevant, a summary line shows the omitted count."""
        nodes = [self._make_node(f"Node about topic {i}") for i in range(12)]
        cluster = [_make_entry(focus="Unrelated focus query")]
        text = _format_existing_knowledge(nodes, cluster=cluster, max_relevant=5)
        self.assertIn("7 more active nodes", text)

    def test_fills_with_high_confidence(self):
        """When fewer than max_relevant nodes have keyword overlap,
        remaining slots are filled with highest-confidence nodes."""
        relevant = self._make_node("Swift adapter training pipeline", confidence=0.3)
        high_conf = self._make_node("Docker compose for Postgres setup", confidence=0.9)
        low_conf = self._make_node("Random note about nothing", confidence=0.1)
        cluster = [_make_entry(focus="Training swift adapter")]
        text = _format_existing_knowledge(
            [low_conf, high_conf, relevant], cluster=cluster, max_relevant=8,
        )
        # All 3 fit in max_relevant=8, but relevant appears first
        self.assertIn("Swift adapter", text)
        self.assertIn("Docker compose", text)
        # relevant node first (has keyword overlap), then high_conf (higher confidence)
        self.assertLess(text.index("Swift adapter"), text.index("Docker compose"))
        self.assertLess(text.index("Docker compose"), text.index("Random note"))

    def test_max_relevant_cap(self):
        """Only max_relevant nodes appear even when all have some overlap."""
        nodes = [self._make_node(f"Swift adapter v{i}") for i in range(10)]
        cluster = [_make_entry(focus="Swift adapter work")]
        text = _format_existing_knowledge(nodes, cluster=cluster, max_relevant=3)
        # Only 3 nodes + summary line
        lines = [l for l in text.split("\n") if l.strip()]
        # 3 node lines + 1 summary = 4 lines
        self.assertEqual(len(lines), 4)
        self.assertIn("7 more active nodes", text)


class TestEstimateResponseBudget(unittest.TestCase):
    """Tests for dynamic response token budget estimation."""

    def test_short_prompt_gets_full_budget(self):
        """Short prompt → large budget (CONTEXT_BUDGET - prompt_tokens)."""
        prompt = "x" * 2000  # ~500 tokens
        budget = _estimate_response_budget(prompt)
        self.assertEqual(budget, CONTEXT_BUDGET - 500)

    def test_long_prompt_gets_minimum(self):
        """Very long prompt → clamped to MIN_RESPONSE_TOKENS."""
        prompt = "x" * 40000  # ~10000 tokens, exceeds CONTEXT_BUDGET
        budget = _estimate_response_budget(prompt)
        self.assertEqual(budget, MIN_RESPONSE_TOKENS)

    def test_never_below_minimum(self):
        """Budget never drops below MIN_RESPONSE_TOKENS regardless of prompt size."""
        for chars in [0, 1000, 10000, 50000, 100000]:
            budget = _estimate_response_budget("x" * chars)
            self.assertGreaterEqual(budget, MIN_RESPONSE_TOKENS)


class TestMinimalPrompt(unittest.TestCase):
    """Tests for adapter-aware minimal prompt selection."""

    def test_minimal_prompt_with_adapter(self):
        """When adapter_path is set, uses minimal prompt without rules/examples."""
        entry = _make_entry(focus="Working on recall")
        node = KnowledgeNode.create(content="Use MLX locally", category="tooling")
        prompt = _build_consolidation_prompt(
            [entry], [node], adapter_path="/some/adapter",
        )
        # Minimal prompt should NOT have verbose rules or BAD examples
        self.assertNotIn("NEVER produce these", prompt)
        self.assertNotIn("BAD knowledge nodes", prompt)
        self.assertNotIn("Rules:", prompt)
        # But SHOULD have data sections
        self.assertIn("Project Context", prompt)
        self.assertIn("Existing Knowledge", prompt)
        self.assertIn("Recent Sessions", prompt)
        self.assertIn("Use MLX locally", prompt)

    def test_full_prompt_without_adapter(self):
        """Without adapter_path, uses full prompt with rules and examples."""
        entry = _make_entry(focus="Working on recall")
        node = KnowledgeNode.create(content="Use MLX locally", category="tooling")
        prompt = _build_consolidation_prompt([entry], [node])
        self.assertIn("NEVER produce these", prompt)
        self.assertIn("Rules:", prompt)
        self.assertIn("Existing Knowledge", prompt)

    def test_minimal_prompt_shorter(self):
        """Minimal prompt should be shorter than full prompt for same inputs."""
        entry = _make_entry(focus="Working on recall search")
        node = KnowledgeNode.create(content="Use MLX locally", category="tooling")
        full = _build_consolidation_prompt([entry], [node])
        minimal = _build_consolidation_prompt(
            [entry], [node], adapter_path="/adapter",
        )
        self.assertLess(len(minimal), len(full))

    def test_minimal_prompt_has_categories(self):
        """Minimal prompt includes the category enum so the model knows valid values."""
        entry = _make_entry(focus="Working on recall")
        node = KnowledgeNode.create(content="Use MLX locally", category="tooling")
        prompt = _build_consolidation_prompt(
            [entry], [node], adapter_path="/some/adapter",
        )
        self.assertIn("Categories:", prompt)
        for cat in ["workflow", "architecture", "debugging", "convention", "tooling"]:
            self.assertIn(cat, prompt)


# ---------------------------------------------------------------------------
# Agent-aware consolidation (#116 steps 3+4)
# ---------------------------------------------------------------------------

class TestAgentAwareConsolidation(unittest.TestCase):
    """Test concurrent agent detection and prompt annotation."""

    def _make_entry(self, session_id="s1", timestamp="2026-03-17T10:00:00Z",
                    griptree="", agent_id="", focus="", done=None):
        return JournalEntry(
            timestamp=timestamp,
            session_id=session_id,
            griptree=griptree,
            agent_id=agent_id,
            focus=focus,
            done=done or [],
        )

    def test_concurrent_agents_detected(self):
        """Two agents with overlapping timestamps are detected as concurrent."""
        from synapt.recall.consolidate import _detect_concurrent_agents
        entries = [
            self._make_entry("s1", "2026-03-17T10:00:00Z", griptree="synapt/main"),
            self._make_entry("s2", "2026-03-17T10:15:00Z", griptree="synapt/feature"),
        ]
        note = _detect_concurrent_agents(entries)
        self.assertIn("CONCURRENT", note)
        self.assertIn("synapt/main", note)
        self.assertIn("synapt/feature", note)

    def test_same_agent_not_flagged(self):
        """Two sessions from the same agent are not flagged as concurrent."""
        from synapt.recall.consolidate import _detect_concurrent_agents
        entries = [
            self._make_entry("s1", "2026-03-17T10:00:00Z", griptree="synapt/main"),
            self._make_entry("s2", "2026-03-17T10:15:00Z", griptree="synapt/main"),
        ]
        note = _detect_concurrent_agents(entries)
        self.assertEqual(note, "")

    def test_no_griptree_not_flagged(self):
        """Entries without griptree metadata are not flagged."""
        from synapt.recall.consolidate import _detect_concurrent_agents
        entries = [
            self._make_entry("s1", "2026-03-17T10:00:00Z"),
            self._make_entry("s2", "2026-03-17T10:15:00Z"),
        ]
        note = _detect_concurrent_agents(entries)
        self.assertEqual(note, "")

    def test_non_overlapping_agents_not_flagged(self):
        """Two agents hours apart are not flagged as concurrent."""
        from synapt.recall.consolidate import _detect_concurrent_agents
        entries = [
            self._make_entry("s1", "2026-03-17T08:00:00Z", griptree="synapt/main"),
            self._make_entry("s2", "2026-03-17T14:00:00Z", griptree="synapt/feature"),
        ]
        note = _detect_concurrent_agents(entries)
        self.assertEqual(note, "")

    def test_format_includes_agent_identity(self):
        """Formatted cluster includes agent griptree labels."""
        from synapt.recall.consolidate import _format_journal_cluster
        entries = [
            self._make_entry("s1", "2026-03-17T10:00:00Z", griptree="synapt/main",
                             focus="fix auth bug"),
        ]
        text = _format_journal_cluster(entries)
        self.assertIn("synapt/main", text)
        self.assertIn("fix auth bug", text)

    def test_format_includes_concurrency_note(self):
        """Formatted cluster of concurrent agents includes annotation."""
        from synapt.recall.consolidate import _format_journal_cluster
        entries = [
            self._make_entry("s1", "2026-03-17T10:00:00Z", griptree="synapt/main",
                             focus="fix auth"),
            self._make_entry("s2", "2026-03-17T10:10:00Z", griptree="synapt/feature",
                             focus="add tests"),
        ]
        text = _format_journal_cluster(entries)
        self.assertIn("CONCURRENT", text)
        self.assertIn("collaborative", text)


# ---------------------------------------------------------------------------
# Temporal date validation (#158 follow-up)
# ---------------------------------------------------------------------------

class TestValidateIsoDate(unittest.TestCase):
    """Test _validate_iso_date for LLM output sanitization."""

    def test_valid_date(self):
        from synapt.recall.consolidate import _validate_iso_date
        self.assertEqual(_validate_iso_date("2026-03-15"), "2026-03-15")

    def test_valid_timestamp(self):
        from synapt.recall.consolidate import _validate_iso_date
        self.assertEqual(
            _validate_iso_date("2026-03-15T10:00:00Z"),
            "2026-03-15T10:00:00Z",
        )

    def test_invalid_natural_language(self):
        from synapt.recall.consolidate import _validate_iso_date
        self.assertIsNone(_validate_iso_date("March 2026"))
        self.assertIsNone(_validate_iso_date("soon"))
        self.assertIsNone(_validate_iso_date("last week"))

    def test_none_and_empty(self):
        from synapt.recall.consolidate import _validate_iso_date
        self.assertIsNone(_validate_iso_date(None))
        self.assertIsNone(_validate_iso_date(""))

    def test_non_string(self):
        from synapt.recall.consolidate import _validate_iso_date
        self.assertIsNone(_validate_iso_date(42))


def test_extract_collections_can_be_disabled(tmp_path, monkeypatch):
    from synapt.recall.consolidate import extract_collections

    monkeypatch.setenv("SYNAPT_DISABLE_ENTITY_COLLECTION", "1")
    assert extract_collections(tmp_path) == 0


# ---------------------------------------------------------------------------
# Tests: _find_best_span threshold and offset resolution coverage
# ---------------------------------------------------------------------------

def test_find_best_span_single_keyword_match():
    """_find_best_span now resolves with a single keyword overlap."""
    from synapt.recall.consolidate import _find_best_span

    node_text = "adoption"
    chunk_text = (
        "We talked about the weather. "
        "She mentioned her adoption plans were moving forward. "
        "Then we discussed dinner."
    )
    span = _find_best_span(node_text, chunk_text)
    assert span is not None
    begin, end = span
    snippet = chunk_text[begin:end]
    assert "adoption" in snippet


def test_find_best_span_no_overlap_returns_none():
    """_find_best_span returns None when there is zero keyword overlap."""
    from synapt.recall.consolidate import _find_best_span

    span = _find_best_span("kubernetes", "We had lunch and discussed movies.")
    assert span is None


def test_find_best_span_empty_inputs():
    """_find_best_span returns None for empty inputs."""
    from synapt.recall.consolidate import _find_best_span

    assert _find_best_span("", "some text") is None
    assert _find_best_span("query", "") is None


# ---------------------------------------------------------------------------
# Tests: content-type-aware dedup thresholds (#337)
# ---------------------------------------------------------------------------

def test_get_dedup_thresholds_personal():
    """Personal content gets higher thresholds (more permissive)."""
    from synapt.recall.consolidate import _get_dedup_thresholds

    class FakeProfile:
        content_type = "personal"

    j, c = _get_dedup_thresholds(FakeProfile())
    assert j == 0.6, f"Expected Jaccard 0.6, got {j}"
    assert c == 0.88, f"Expected cosine 0.88, got {c}"


def test_get_dedup_thresholds_code():
    """Code content gets lower thresholds (more aggressive dedup)."""
    from synapt.recall.consolidate import _get_dedup_thresholds

    class FakeProfile:
        content_type = "code"

    j, c = _get_dedup_thresholds(FakeProfile())
    assert j == 0.4, f"Expected Jaccard 0.4, got {j}"
    assert c == 0.75, f"Expected cosine 0.75, got {c}"


def test_get_dedup_thresholds_mixed_default():
    """Mixed/unknown content gets default thresholds."""
    from synapt.recall.consolidate import _get_dedup_thresholds

    class FakeProfile:
        content_type = "mixed"

    j, c = _get_dedup_thresholds(FakeProfile())
    assert j == 0.5
    assert c == 0.80


def test_get_dedup_thresholds_none_fallback():
    """None content_profile falls back to mixed thresholds."""
    from synapt.recall.consolidate import _get_dedup_thresholds

    j, c = _get_dedup_thresholds(None)
    assert j == 0.5
    assert c == 0.80


def test_get_dedup_thresholds_unknown_type():
    """Unknown content type falls back to mixed thresholds."""
    from synapt.recall.consolidate import _get_dedup_thresholds

    class FakeProfile:
        content_type = "unknown_type"

    j, c = _get_dedup_thresholds(FakeProfile())
    assert j == 0.5
    assert c == 0.80


def _setup_resolve_test(tmp_path, monkeypatch, transcript_path):
    """Helper: set up DB + knowledge node for resolve_source_offsets tests."""
    from synapt.recall.storage import RecallDB
    from synapt.recall.knowledge import KnowledgeNode, append_node

    recall_dir = tmp_path / ".synapt" / "recall"
    recall_dir.mkdir(parents=True)
    index_dir = recall_dir / "index"
    index_dir.mkdir()

    monkeypatch.setattr(
        "synapt.recall.consolidate.project_index_dir",
        lambda _: index_dir,
    )
    kn_path = recall_dir / "knowledge.jsonl"
    monkeypatch.setattr(
        "synapt.recall.consolidate._knowledge_path",
        lambda _: kn_path,
    )

    node = KnowledgeNode.create(
        content="She mentioned her adoption plans",
        category="personal",
        source_sessions=["sess-A"],
        source_turns=["sess-A:0"],
    )
    append_node(node, kn_path)

    db = RecallDB(index_dir / "recall.db")
    db._conn.execute(
        "INSERT INTO chunks (id, session_id, timestamp, turn_index, user_text, assistant_text, transcript_path) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("chunk-001", "sess-A", "2026-03-01T00:00:00", 0,
         "We talked about the weather. She mentioned her adoption plans were moving forward. Then we discussed dinner.",
         "",
         transcript_path),
    )
    db._conn.commit()
    db.close()
    return kn_path


def test_resolve_source_offsets_includes_file_path(tmp_path, monkeypatch):
    """resolve_source_offsets stores transcript_path as 'f' key in offset dicts."""
    from synapt.recall.consolidate import resolve_source_offsets

    kn_path = _setup_resolve_test(tmp_path, monkeypatch, "/transcripts/sess-A.jsonl")

    count = resolve_source_offsets(tmp_path)
    assert count == 1

    nodes = list(read_nodes(kn_path))
    resolved = nodes[0]
    assert resolved.source_offsets, "source_offsets should be populated"
    offset = resolved.source_offsets[0]
    assert offset["s"] == "sess-A"
    assert offset["t"] == 0
    assert "b" in offset and "e" in offset
    assert offset["f"] == "/transcripts/sess-A.jsonl"
    # Verify the span actually contains the right text
    chunk_text = "We talked about the weather. She mentioned her adoption plans were moving forward. Then we discussed dinner."
    snippet = chunk_text[offset["b"]:offset["e"]]
    assert "adoption" in snippet


def test_resolve_source_offsets_omits_f_when_empty(tmp_path, monkeypatch):
    """When transcript_path is empty, the 'f' key is omitted from offsets."""
    from synapt.recall.consolidate import resolve_source_offsets

    kn_path = _setup_resolve_test(tmp_path, monkeypatch, "")

    count = resolve_source_offsets(tmp_path)
    assert count == 1

    nodes = list(read_nodes(kn_path))
    offset = nodes[0].source_offsets[0]
    assert "f" not in offset, "Empty transcript_path should not be stored"


# ---------------------------------------------------------------------------
# recall#865 — wire extract into recall (consolidation slot, SYNAPT_USE_EXTRACT)
#
# Spec: config/design/move-1-extract-into-recall-2026-07-12.md
# Contract-read: #dev, 2026-07-12 (field mapping, MLX-as-Stage-1 design,
# hash-anchoring via source_id).
# ---------------------------------------------------------------------------


class TestSynaptExtractDependencyImports(unittest.TestCase):
    """The new dependency must import cleanly (spec acceptance criterion).

    Imports exactly what consolidate.py imports -- validate_extraction is
    called internally by finalize_extraction, not directly by the impl, so
    it's deliberately left out here (Opus review m_13ae8e8a, nit)."""

    def test_synapt_extract_imports_cleanly(self):
        import synapt_extract  # noqa: F401
        from synapt_extract import (  # noqa: F401
            create_extraction_builder,
            finalize_extraction,
            FinalizeContext,
        )


class TestExtractSourceId(unittest.TestCase):
    """_extract_source_id: content-derived hash so the packet is addressable
    (the "hash-anchored" acceptance criterion — not automatic from extract
    itself, per the contract-read)."""

    def test_deterministic_for_same_content(self):
        self.assertEqual(_extract_source_id("hello world"), _extract_source_id("hello world"))

    def test_differs_for_different_content(self):
        self.assertNotEqual(_extract_source_id("hello world"), _extract_source_id("goodbye world"))

    def test_matches_sha256_prefix(self):
        text = "the loom weaves context"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        self.assertEqual(_extract_source_id(text), expected)


class TestMapExtractionCategory(unittest.TestCase):
    """extract's fact.category is free-form text, not constrained to recall's
    VALID_CATEGORIES enum — map through with a "fact" fallback."""

    def test_passthrough_for_valid_category(self):
        self.assertEqual(_map_extraction_category("architecture"), "architecture")
        self.assertEqual(_map_extraction_category("preference"), "preference")

    def test_defaults_to_fact_for_unknown_category(self):
        self.assertEqual(_map_extraction_category("random_unmapped_label"), "fact")

    def test_defaults_to_fact_for_missing_category(self):
        self.assertEqual(_map_extraction_category(None), "fact")
        self.assertEqual(_map_extraction_category(""), "fact")


class TestEarliestTemporalRefDate(unittest.TestCase):
    """temporal_refs isn't per-fact-linked without the evidence_anchoring
    capability (more prompt surface than this slot requests) — the earliest
    resolved date sets valid_from for all nodes from the packet, same shape
    as today's _cluster_valid_from() fallback."""

    def test_empty_list_returns_none(self):
        self.assertIsNone(_earliest_temporal_ref_date([]))

    def test_single_resolved_ref(self):
        refs = [{"raw": "March 2026", "resolved": "2026-03-01"}]
        self.assertEqual(_earliest_temporal_ref_date(refs), "2026-03-01")

    def test_picks_earliest_of_multiple(self):
        refs = [
            {"raw": "April 2026", "resolved": "2026-04-01"},
            {"raw": "January 2026", "resolved": "2026-01-15"},
            {"raw": "March 2026", "resolved": "2026-03-01"},
        ]
        self.assertEqual(_earliest_temporal_ref_date(refs), "2026-01-15")

    def test_skips_refs_missing_resolved(self):
        refs = [
            {"raw": "sometime later"},
            {"raw": "March 2026", "resolved": "2026-03-01"},
        ]
        self.assertEqual(_earliest_temporal_ref_date(refs), "2026-03-01")

    def test_skips_unparseable_dates(self):
        refs = [
            {"raw": "garbled", "resolved": "not-a-date"},
            {"raw": "March 2026", "resolved": "2026-03-01"},
        ]
        self.assertEqual(_earliest_temporal_ref_date(refs), "2026-03-01")

    def test_all_unparseable_returns_none(self):
        self.assertIsNone(_earliest_temporal_ref_date([{"raw": "garbled", "resolved": "not-a-date"}]))


class TestKnowledgeNodesFromExtraction(unittest.TestCase):
    """Core field mapping: SynaptExtraction packet (facts/decisions/
    temporal_refs) -> KnowledgeNode list. entities are NOT materialized as
    nodes in this slot (structural, not durable prose) — see contract-read."""

    def _make_packet(self, **overrides):
        packet = {
            "version": "1",
            "extracted_at": "2026-07-12T00:00:00Z",
            "produced_by": "mlx://mlx-community/Ministral-3-3B-Instruct-2512-4bit",
            "source_id": "abc123def4567890",
            "entities": [],
            "goals": [],
            "themes": [],
            "facts": [
                {"text": "recall does not use extract as of 2026-07-12", "category": "architecture"},
                {"text": "Layne prefers semicolons over em dashes"},
            ],
            "decisions": [
                {"text": "Move 1 targets the consolidation slot, not enrich.py"},
            ],
            "temporal_refs": [
                {"raw": "2026-07-12", "resolved": "2026-07-12"},
            ],
            "capabilities": ["facts", "decisions", "temporal_refs"],
        }
        packet.update(overrides)
        return packet

    def test_creates_one_node_per_fact_and_decision(self):
        nodes = _knowledge_nodes_from_extraction(self._make_packet(), ["s1", "s2"])
        self.assertEqual(len(nodes), 3)

    def test_fact_content_matches_text(self):
        nodes = _knowledge_nodes_from_extraction(self._make_packet(), ["s1"])
        contents = {n.content for n in nodes}
        self.assertIn("recall does not use extract as of 2026-07-12", contents)

    def test_fact_category_mapped_when_valid(self):
        nodes = _knowledge_nodes_from_extraction(self._make_packet(), ["s1"])
        node = next(n for n in nodes if n.content.startswith("recall does not use extract"))
        self.assertEqual(node.category, "architecture")

    def test_fact_category_defaults_to_fact_when_missing(self):
        nodes = _knowledge_nodes_from_extraction(self._make_packet(), ["s1"])
        node = next(n for n in nodes if n.content.startswith("Layne prefers semicolons"))
        self.assertEqual(node.category, "fact")

    def test_decision_category_hardcoded(self):
        nodes = _knowledge_nodes_from_extraction(self._make_packet(), ["s1"])
        node = next(n for n in nodes if n.content.startswith("Move 1 targets"))
        self.assertEqual(node.category, "decision")

    def test_confidence_uses_recall_compute_confidence(self):
        cluster_sessions = ["s1", "s2", "s3"]
        nodes = _knowledge_nodes_from_extraction(self._make_packet(), cluster_sessions)
        expected = compute_confidence(len(cluster_sessions))
        for n in nodes:
            self.assertAlmostEqual(n.confidence, expected)

    def test_source_sessions_populated_from_cluster(self):
        nodes = _knowledge_nodes_from_extraction(self._make_packet(), ["s1", "s2"])
        for n in nodes:
            self.assertEqual(n.source_sessions, ["s1", "s2"])

    def test_valid_from_uses_earliest_temporal_ref(self):
        nodes = _knowledge_nodes_from_extraction(self._make_packet(), ["s1"])
        for n in nodes:
            self.assertEqual(n.valid_from, "2026-07-12")

    def test_empty_facts_and_decisions_produce_no_nodes(self):
        nodes = _knowledge_nodes_from_extraction(self._make_packet(facts=[], decisions=[]), ["s1"])
        self.assertEqual(nodes, [])

    def test_entities_are_not_materialized_as_nodes(self):
        packet = self._make_packet(entities=[{"name": "recall", "type": "system"}])
        nodes = _knowledge_nodes_from_extraction(packet, ["s1"])
        self.assertEqual(len(nodes), 3)  # unchanged: 2 facts + 1 decision, no entity node

    def test_returns_knowledge_node_instances(self):
        nodes = _knowledge_nodes_from_extraction(self._make_packet(), ["s1"])
        for n in nodes:
            self.assertIsInstance(n, KnowledgeNode)
            self.assertTrue(n.id)
            self.assertTrue(n.created_at)


class _FakeExtractionClient:
    """Duck-types synapt._models.base.ModelClient for extraction tests —
    no real MLX runtime required."""

    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    def chat(self, model, messages, temperature=0.1, adapter_path=None, max_tokens=None, **kwargs):
        self.calls.append({
            "model": model,
            "prompt": messages[0].content if messages else "",
            "temperature": temperature,
            "adapter_path": adapter_path,
            "max_tokens": max_tokens,
        })
        return self.response_text


class TestBuildExtractionPacket(unittest.TestCase):
    """_build_extraction_packet: extract Stage 1 (via *client*) -> finalize
    -> validate, synchronously. Bypasses extract.py's async orchestrator
    (consolidate.py is sync; MLXClient.chat() is blocking) — see contract-read
    point 2. Returns None (fail-closed) on unparseable or schema-invalid
    output, mirroring today's unparseable-response handling."""

    VALID_STAGE1_JSON = json.dumps({
        "extracted_at": "2026-07-12T00:00:00Z",
        "facts": [{"text": "recall does not use extract as of 2026-07-12", "category": "architecture"}],
        "decisions": [{"text": "Move 1 targets the consolidation slot"}],
        "temporal_refs": [{"raw": "2026-07-12", "resolved": "2026-07-12"}],
    })

    def test_valid_response_produces_validated_packet(self):
        client = _FakeExtractionClient(self.VALID_STAGE1_JSON)
        packet = _build_extraction_packet("some cluster text", client, model="mlx-community/test-model")
        self.assertIsNotNone(packet)
        self.assertEqual(packet["version"], "1")
        self.assertEqual(len(packet["facts"]), 1)
        self.assertEqual(len(packet["decisions"]), 1)

    def test_packet_is_hash_anchored(self):
        client = _FakeExtractionClient(self.VALID_STAGE1_JSON)
        text = "some cluster text"
        packet = _build_extraction_packet(text, client, model="mlx-community/test-model")
        self.assertEqual(packet["source_id"], _extract_source_id(text))

    def test_packet_produced_by_is_mlx_uri(self):
        client = _FakeExtractionClient(self.VALID_STAGE1_JSON)
        packet = _build_extraction_packet("some cluster text", client, model="mlx-community/test-model")
        self.assertEqual(packet["produced_by"], "mlx://mlx-community/test-model")

    def test_calls_client_with_extraction_prompt(self):
        client = _FakeExtractionClient(self.VALID_STAGE1_JSON)
        _build_extraction_packet("some cluster text", client, model="mlx-community/test-model")
        self.assertEqual(len(client.calls), 1)
        self.assertIn("some cluster text", client.calls[0]["prompt"])

    def test_threads_adapter_path_and_max_tokens_to_client(self):
        client = _FakeExtractionClient(self.VALID_STAGE1_JSON)
        _build_extraction_packet(
            "some cluster text", client, model="mlx-community/test-model",
            adapter_path="/path/to/adapter", max_tokens=1234,
        )
        call = client.calls[0]
        self.assertEqual(call["adapter_path"], "/path/to/adapter")
        self.assertEqual(call["max_tokens"], 1234)

    def test_unparseable_response_returns_none(self):
        client = _FakeExtractionClient("this is not json at all {{{")
        packet = _build_extraction_packet("some cluster text", client, model="mlx-community/test-model")
        self.assertIsNone(packet)

    def test_schema_invalid_response_returns_none(self):
        # Fact missing required "text" -- fails extract's structural validation.
        bad_json = json.dumps({
            "extracted_at": "2026-07-12T00:00:00Z",
            "facts": [{"category": "architecture"}],
            "decisions": [],
            "temporal_refs": [],
        })
        client = _FakeExtractionClient(bad_json)
        packet = _build_extraction_packet("some cluster text", client, model="mlx-community/test-model")
        self.assertIsNone(packet)

    def test_backfills_missing_entities_goals_themes(self):
        # extract's finalized schema unconditionally requires entities/goals/
        # themes even though the Stage-1 request schema only requires the
        # capabilities we asked for (facts/decisions/temporal_refs) -- a real
        # gap found writing these tests, not covered by the original
        # contract-read. _build_extraction_packet must backfill empty arrays
        # so validation passes.
        client = _FakeExtractionClient(self.VALID_STAGE1_JSON)
        packet = _build_extraction_packet("some cluster text", client, model="mlx-community/test-model")
        self.assertIsNotNone(packet)
        self.assertEqual(packet["entities"], [])
        self.assertEqual(packet["goals"], [])
        self.assertEqual(packet["themes"], [])

    def test_default_budget_scales_with_prompt_not_flat(self):
        # Opus review (m_13ae8e8a, blocker 2): a flat low budget silently
        # truncates dense clusters mid-JSON. Default (no max_tokens passed)
        # must scale like legacy's _estimate_response_budget(), not sit at
        # a fixed MIN_RESPONSE_TOKENS regardless of prompt size.
        client = _FakeExtractionClient(self.VALID_STAGE1_JSON)
        _build_extraction_packet("some cluster text", client, model="mlx-community/test-model")
        sent_prompt = client.calls[0]["prompt"]
        self.assertEqual(client.calls[0]["max_tokens"], _estimate_response_budget(sent_prompt))

    def test_default_budget_shrinks_for_longer_prompt(self):
        # Same shape as TestEstimateResponseBudget -- proves the extraction
        # call site actually varies the budget with input size, not just
        # matching _estimate_response_budget's return value coincidentally
        # at one fixed input length.
        short_client = _FakeExtractionClient(self.VALID_STAGE1_JSON)
        _build_extraction_packet("short cluster text", short_client, model="mlx-community/test-model")

        long_client = _FakeExtractionClient(self.VALID_STAGE1_JSON)
        _build_extraction_packet("dense cluster text " * 200, long_client, model="mlx-community/test-model")

        self.assertLess(long_client.calls[0]["max_tokens"], short_client.calls[0]["max_tokens"])

    def test_explicit_max_tokens_still_overrides_default(self):
        client = _FakeExtractionClient(self.VALID_STAGE1_JSON)
        _build_extraction_packet(
            "dense cluster text " * 200, client, model="mlx-community/test-model",
            max_tokens=42,
        )
        self.assertEqual(client.calls[0]["max_tokens"], 42)

    def test_requests_exactly_the_pinned_capabilities(self):
        # Nit (Opus review m_13ae8e8a): pin what's actually requested from
        # extract so capability creep (accidentally widening the Stage-1
        # prompt surface for the fragile 3B model) shows up as a failing
        # test, not a silent diff.
        self.assertEqual(list(EXTRACTION_CAPABILITIES), ["facts", "decisions", "temporal_refs"])

    def test_client_exception_returns_none(self):
        class _RaisingClient:
            def chat(self, *args, **kwargs):
                raise RuntimeError("inference backend unavailable")

        packet = _build_extraction_packet("some cluster text", _RaisingClient(), model="mlx-community/test-model")
        self.assertIsNone(packet)

    # Sentinel review (m_40b29111, HIGH blocker): parseable Stage-1 JSON can
    # still have non-dict members inside facts/decisions/temporal_refs --
    # the local MLX client has no schema-constrained decoding. Against real
    # synapt_extract 0.5.0, finalize_extraction() -> _detect_capabilities()
    # calls .get() on each array member without an isinstance guard, so
    # each of these three shapes raises AttributeError instead of returning
    # an invalid ValidationResult (confirmed directly against the installed
    # package before writing these tests, not assumed from the review).
    # These go through _build_extraction_packet() end-to-end -- the earlier
    # per-item-guard tests in TestKnowledgeNodesFromExtractionGuards call
    # _knowledge_nodes_from_extraction() directly with a hand-built packet
    # that could never survive finalization, so they never exercised this
    # boundary. Fail-closed here means None, not a raised exception.

    def test_null_fact_member_does_not_crash(self):
        bad_json = json.dumps({
            "extracted_at": "2026-07-12T00:00:00Z",
            "facts": [None],
            "decisions": [],
            "temporal_refs": [],
        })
        client = _FakeExtractionClient(bad_json)
        packet = _build_extraction_packet("some cluster text", client, model="mlx-community/test-model")
        self.assertIsNone(packet)

    def test_non_dict_decision_member_does_not_crash(self):
        bad_json = json.dumps({
            "extracted_at": "2026-07-12T00:00:00Z",
            "facts": [],
            "decisions": [42],
            "temporal_refs": [],
        })
        client = _FakeExtractionClient(bad_json)
        packet = _build_extraction_packet("some cluster text", client, model="mlx-community/test-model")
        self.assertIsNone(packet)

    def test_string_temporal_ref_member_does_not_crash(self):
        bad_json = json.dumps({
            "extracted_at": "2026-07-12T00:00:00Z",
            "facts": [],
            "decisions": [],
            "temporal_refs": ["tomorrow"],
        })
        client = _FakeExtractionClient(bad_json)
        packet = _build_extraction_packet("some cluster text", client, model="mlx-community/test-model")
        self.assertIsNone(packet)


class TestKnowledgeNodesFromExtractionGuards(unittest.TestCase):
    """Per-item guards (Opus review m_13ae8e8a, nit): malformed items within
    facts/decisions must be skipped, not crash the whole packet mapping."""

    def test_non_dict_fact_item_skipped(self):
        packet = {
            "facts": [None, "not a dict", {"text": "a real fact"}],
            "decisions": [],
            "temporal_refs": [],
        }
        nodes = _knowledge_nodes_from_extraction(packet, ["s1"])
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].content, "a real fact")

    def test_non_dict_decision_item_skipped(self):
        packet = {
            "facts": [],
            "decisions": [None, 42, {"text": "a real decision"}],
            "temporal_refs": [],
        }
        nodes = _knowledge_nodes_from_extraction(packet, ["s1"])
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].content, "a real decision")

    def test_fact_with_empty_text_skipped(self):
        packet = {
            "facts": [{"text": ""}, {"text": "a real fact"}],
            "decisions": [],
            "temporal_refs": [],
        }
        nodes = _knowledge_nodes_from_extraction(packet, ["s1"])
        self.assertEqual(len(nodes), 1)

    def test_decision_with_missing_text_skipped(self):
        packet = {
            "facts": [],
            "decisions": [{"category": "architecture"}, {"text": "a real decision"}],
            "temporal_refs": [],
        }
        nodes = _knowledge_nodes_from_extraction(packet, ["s1"])
        self.assertEqual(len(nodes), 1)


class TestConsolidateLegacyPathUnchangedWhenFlagOff(unittest.TestCase):
    """Opus review (m_13ae8e8a, medium 5): "legacy byte-identical when flag
    off" was a named acceptance bullet with zero executed test -- nothing
    called consolidate()/_process_cluster anywhere in the suite. This drives
    the real consolidate() entrypoint and proves the extract-path function
    is never invoked when SYNAPT_USE_EXTRACT is unset."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.project_dir = Path(self.tmpdir)
        journal_dir = self.project_dir / ".synapt" / "recall"
        journal_dir.mkdir(parents=True, exist_ok=True)
        entries = [
            {
                "timestamp": "2026-07-12T09:00:00", "session_id": "guard-s1",
                "focus": "Legacy-path guard test session one",
                "done": ["Did something specific in src/synapt/recall/consolidate.py"],
                "decisions": [], "next_steps": [], "files_modified": ["src/synapt/recall/consolidate.py"],
                "enriched": True,
            },
            {
                "timestamp": "2026-07-12T09:30:00", "session_id": "guard-s2",
                "focus": "Legacy-path guard test session two",
                "done": ["Did something else specific in src/synapt/recall/consolidate.py"],
                "decisions": [], "next_steps": [], "files_modified": ["src/synapt/recall/consolidate.py"],
                "enriched": True,
            },
        ]
        with open(journal_dir / "journal.jsonl", "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    def test_flag_off_never_calls_build_extraction_packet(self):
        with patch("synapt.recall.consolidate._build_extraction_packet") as mock_extract, \
             patch("synapt.recall._model_router.get_client", return_value=_FakeExtractionClient(
                 json.dumps({"nodes": []})
             )), \
             patch.dict("os.environ", {"SYNAPT_USE_EXTRACT": ""}):
            consolidate(project_dir=self.project_dir, model="fake-model", min_entries=2)
        mock_extract.assert_not_called()

    def test_flag_unset_never_calls_build_extraction_packet(self):
        # Same as above but SYNAPT_USE_EXTRACT absent entirely, not just
        # empty-string -- both must resolve to "off" per _env_flag.
        import os
        env_without_flag = {k: v for k, v in os.environ.items() if k != "SYNAPT_USE_EXTRACT"}
        with patch("synapt.recall.consolidate._build_extraction_packet") as mock_extract, \
             patch("synapt.recall._model_router.get_client", return_value=_FakeExtractionClient(
                 json.dumps({"nodes": []})
             )), \
             patch.dict("os.environ", env_without_flag, clear=True):
            consolidate(project_dir=self.project_dir, model="fake-model", min_entries=2)
        mock_extract.assert_not_called()


class TestConsolidateFlagOnMalformedMlxOutputDoesNotCrash(unittest.TestCase):
    """Sentinel review (m_40b29111, HIGH blocker): the real bug was that
    structurally malformed Stage-1 JSON crashed the whole consolidate() run
    instead of taking the documented None -> False -> skip/retry path.
    Drives the real flag-on consolidate() entrypoint with each of the three
    confirmed-crashing shapes and asserts the run completes gracefully."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.project_dir = Path(self.tmpdir)
        journal_dir = self.project_dir / ".synapt" / "recall"
        journal_dir.mkdir(parents=True, exist_ok=True)
        entries = [
            {
                "timestamp": "2026-07-12T09:00:00", "session_id": "malformed-s1",
                "focus": "Flag-on malformed-output lifecycle test session one",
                "done": ["Did something specific in src/synapt/recall/consolidate.py"],
                "decisions": [], "next_steps": [], "files_modified": ["src/synapt/recall/consolidate.py"],
                "enriched": True,
            },
            {
                "timestamp": "2026-07-12T09:30:00", "session_id": "malformed-s2",
                "focus": "Flag-on malformed-output lifecycle test session two",
                "done": ["Did something else specific in src/synapt/recall/consolidate.py"],
                "decisions": [], "next_steps": [], "files_modified": ["src/synapt/recall/consolidate.py"],
                "enriched": True,
            },
        ]
        with open(journal_dir / "journal.jsonl", "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    def _run_with_malformed_response(self, malformed_json):
        client = _FakeExtractionClient(malformed_json)
        with patch("synapt.recall._model_router.get_client", return_value=client), \
             patch.dict("os.environ", {"SYNAPT_USE_EXTRACT": "1"}):
            from synapt.recall.consolidate import consolidate
            return consolidate(project_dir=self.project_dir, model="fake-model", min_entries=2)

    def test_null_fact_member_completes_gracefully(self):
        result = self._run_with_malformed_response(json.dumps({
            "extracted_at": "2026-07-12T00:00:00Z",
            "facts": [None], "decisions": [], "temporal_refs": [],
        }))
        self.assertEqual(result.nodes_created, 0)

    def test_non_dict_decision_member_completes_gracefully(self):
        result = self._run_with_malformed_response(json.dumps({
            "extracted_at": "2026-07-12T00:00:00Z",
            "facts": [], "decisions": [42], "temporal_refs": [],
        }))
        self.assertEqual(result.nodes_created, 0)

    def test_string_temporal_ref_member_completes_gracefully(self):
        result = self._run_with_malformed_response(json.dumps({
            "extracted_at": "2026-07-12T00:00:00Z",
            "facts": [], "decisions": [], "temporal_refs": ["tomorrow"],
        }))
        self.assertEqual(result.nodes_created, 0)


if __name__ == "__main__":
    unittest.main()
