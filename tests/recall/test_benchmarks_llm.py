"""LLM inference benchmarks for recall system (#305).

Benchmarks the MLX-powered operations that dominate build time:
- Model loading (cold start + cached)
- Cluster summary generation (single + batch)
- Consolidation knowledge extraction
- Prompt construction overhead
- Peak memory with model loaded

Requires MLX (Apple Silicon). Skipped automatically on other platforms.

Run:
    pytest tests/recall/test_benchmarks_llm.py -v --benchmark-enable -s
    pytest tests/recall/test_benchmarks_llm.py -v --benchmark-only -s
"""

from __future__ import annotations

import os
import time
import tracemalloc

import pytest

# Skip entire module if MLX is not available.
#
# MLX availability comes from its CANONICAL owner, `synapt.recall._mlx`, and nowhere else.
# This block used to read `_MLX_AVAILABLE` out of `synapt.recall.clustering`, which has
# never defined it -- so the import raised, the broad `except ImportError` swallowed it,
# and the module skipped UNCONDITIONALLY on every platform, including Apple Silicon where
# MLX works. All 16 tests here reported "skipped" and provided no signal anywhere.
#
# The lesson is in the shape, not the typo: a try/except ImportError wrapped around a
# MULTI-NAME import cannot distinguish "the backend is absent" from "one of these symbols
# does not exist." It answers an adjacent question and reports the wrong one confidently.
# So the two concerns are now separate statements: the availability probe may fail softly,
# but the collaborator imports below must fail LOUDLY if a name ever moves again.
from synapt.recall._mlx import MLX_AVAILABLE as _MLX_AVAILABLE, SKIP_REASON as _SKIP_REASON

from synapt._models.mlx_client import MLXClient, MLXOptions
from synapt._models.base import Message
from synapt.recall.clustering import (
    _build_cluster_excerpts,
    generate_llm_summary,
    CLUSTER_SUMMARY_PROMPT,
    DEFAULT_MODEL,
    MAX_SUMMARY_TOKENS,
)

# Availability is necessary but NOT sufficient, and the opt-in is deliberate rather than
# an accident of the repair above.
#
# These are real-model benchmarks: `test_model_load_cold` clears MLXClient._BASE_CACHE to
# force a genuine download-and-load, and the `warm_client` fixture runs live inference.
# The module docstring has always said to run them explicitly (`--benchmark-enable -s`).
#
# Repairing the availability import alone would have switched all 16 on inside the default
# CI sweep, because the macOS runners are arm64 and therefore DO install mlx-lm -- turning
# a predicate fix into a multi-gigabyte model download on three Python versions. Whether
# to spend that in CI is a cost decision on its own evidence, so it is stated here as a
# flag instead of arriving as a side effect. Set SYNAPT_RUN_LLM_BENCHMARKS=1 to run them.
RUN_LLM_BENCHMARKS_ENV = "SYNAPT_RUN_LLM_BENCHMARKS"
_OPTED_IN = os.environ.get(RUN_LLM_BENCHMARKS_ENV, "").strip().lower() not in {
    "", "0", "false", "no", "off",
}

if not _MLX_AVAILABLE:
    # `or` rather than a bare read: _mlx guarantees a non-empty reason whenever it reports
    # unavailable, but a guard that silently degrades to "" when that invariant breaks
    # would stop skipping and run real-model tests with no backend. Do not import an
    # invariant you can cheaply not depend on.
    _BENCHMARK_SKIP_REASON = _SKIP_REASON or "requires the MLX backend (`pip install mlx-lm`)"
elif not _OPTED_IN:
    _BENCHMARK_SKIP_REASON = (
        f"real-model LLM benchmarks are opt-in: set {RUN_LLM_BENCHMARKS_ENV}=1 to run "
        "them (they download and load a model, so they are excluded from the default "
        "suite by choice, not by a broken guard)"
    )
else:
    _BENCHMARK_SKIP_REASON = ""

pytestmark = pytest.mark.skipif(bool(_BENCHMARK_SKIP_REASON), reason=_BENCHMARK_SKIP_REASON)


# ---------------------------------------------------------------------------
# Synthetic cluster data
# ---------------------------------------------------------------------------

def _make_cluster_texts(n_chunks: int) -> list[dict]:
    """Generate synthetic cluster chunk texts for LLM benchmarks."""
    texts = []
    topics = [
        ("fix flock race condition in compact_journal",
         "The flock TOCTOU race was in the stat-before-flock pattern. "
         "Rewrote compact_journal to use advisory locking with retry."),
        ("test flock locking under concurrent access",
         "Added concurrent writer test. Advisory lock prevents double-write. "
         "Edge case: empty journal file now handled gracefully."),
        ("review flock compact rewrite for edge cases",
         "Reviewed compact_journal rewrite. Covers empty file, partial write, "
         "and signal interruption. TOCTOU window eliminated."),
        ("benchmark flock locking performance overhead",
         "Advisory lock adds ~0.2ms per compact call. Acceptable given it "
         "prevents data corruption. WAL mode reduces contention further."),
        ("deploy flock fix and monitor production journal writes",
         "Deployed flock fix. Journal compaction now atomic. Monitoring shows "
         "zero corruption events in 48 hours. Lock contention under 1%."),
        ("optimize flock retry backoff for high contention scenarios",
         "Exponential backoff with jitter reduces lock wait time by 60% under "
         "contention. Max retry of 3 with 10ms base delay."),
        ("add flock telemetry to track lock acquisition latency",
         "Added structured logging for lock acquire/release timing. P99 latency "
         "is 0.8ms, well under the 5ms budget. Histogram shows bimodal distribution."),
    ]
    for i in range(n_chunks):
        user, asst = topics[i % len(topics)]
        texts.append({"user_text": user, "assistant_text": asst})
    return texts


def _make_journal_entries(n: int = 5):
    """Generate synthetic journal entries for consolidation benchmarks."""
    from synapt.recall.journal import JournalEntry

    entries = []
    for i in range(n):
        entries.append(JournalEntry(
            timestamp=f"2026-03-0{i+1}T10:00:00Z",
            session_id=f"session-bench-{i:03d}",
            branch="feat/flock-fix",
            focus=f"Fix flock advisory locking — iteration {i+1}",
            done=[
                f"Implemented retry backoff (attempt {i+1})",
                "Added concurrent writer tests",
                "Reviewed compact_journal edge cases",
            ],
            decisions=[
                "Use advisory locking over mandatory locking",
                "Exponential backoff with jitter for retries",
            ],
            next_steps=["Deploy to production", "Add telemetry"],
            files_modified=["src/synapt/recall/cli.py", "src/synapt/recall/journal.py"],
            enriched=True,
        ))
    return entries


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mlx_client():
    """Create an MLXClient once per module (model stays cached)."""
    return MLXClient(MLXOptions(max_tokens=MAX_SUMMARY_TOKENS))


@pytest.fixture(scope="module")
def warm_client(mlx_client):
    """Ensure model is loaded and cached before benchmarks run."""
    # Warm up: trigger model download/load
    texts = _make_cluster_texts(2)
    generate_llm_summary(texts, "warmup", client=mlx_client)
    return mlx_client


@pytest.fixture
def cluster_texts_5():
    return _make_cluster_texts(5)


@pytest.fixture
def cluster_texts_10():
    return _make_cluster_texts(10)


@pytest.fixture
def cluster_texts_3():
    return _make_cluster_texts(3)


# ═══════════════════════════════════════════════════════════════════════════
# 1. MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

class TestModelLoading:
    """Benchmark model load time — the cold start cost."""

    def test_model_load_cold(self, capsys):
        """Cold load: download check + weight loading + tokenizer init.

        This is the cost paid on first inference after process start.
        Subsequent calls hit the class-level cache.
        """
        # Clear the cache to force a cold load
        MLXClient._BASE_CACHE.clear()
        MLXClient._CURRENT_ADAPTER.clear()

        client = MLXClient(MLXOptions(max_tokens=MAX_SUMMARY_TOKENS))

        start = time.perf_counter()
        # Trigger load by calling chat
        client.chat(
            model=DEFAULT_MODEL,
            messages=[Message(role="user", content="test")],
            temperature=0.1,
        )
        elapsed = time.perf_counter() - start
        print(f"\n  model cold load + first inference: {elapsed:.2f}s")

    def test_model_load_cached(self, benchmark, warm_client):
        """Cached load: model already in memory, just format + generate."""

        def inference():
            return warm_client.chat(
                model=DEFAULT_MODEL,
                messages=[Message(role="user", content="Summarize: test benchmark")],
                temperature=0.1,
            )

        benchmark.pedantic(inference, rounds=1, warmup_rounds=0)


# ═══════════════════════════════════════════════════════════════════════════
# 2. CLUSTER SUMMARY INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

class TestClusterSummaryInference:
    """Benchmark the generate_llm_summary pipeline end-to-end."""

    def test_summary_3_chunks(self, benchmark, warm_client, cluster_texts_3):
        """LLM summary for a small cluster (3 chunks)."""
        benchmark.pedantic(
            generate_llm_summary,
            args=(cluster_texts_3, "flock locking"),
            kwargs={"client": warm_client},
            rounds=1, warmup_rounds=0,
        )

    def test_summary_5_chunks(self, benchmark, warm_client, cluster_texts_5):
        """LLM summary for a medium cluster (5 chunks — the default threshold)."""
        benchmark.pedantic(
            generate_llm_summary,
            args=(cluster_texts_5, "flock locking"),
            kwargs={"client": warm_client},
            rounds=1, warmup_rounds=0,
        )

    def test_summary_10_chunks(self, benchmark, warm_client, cluster_texts_10):
        """LLM summary for a large cluster (10 chunks)."""
        benchmark.pedantic(
            generate_llm_summary,
            args=(cluster_texts_10, "flock locking"),
            kwargs={"client": warm_client},
            rounds=1, warmup_rounds=0,
        )

    def test_batch_5_summaries(self, warm_client, capsys):
        """Simulate upgrade_large_cluster_summaries batch of 5.

        This is the real-world cost: 5 sequential LLM calls during build.
        """
        clusters = [_make_cluster_texts(5 + i) for i in range(5)]
        topics = [
            "flock locking", "jaccard clustering",
            "swift adapter", "recall search", "git rebase",
        ]

        start = time.perf_counter()
        results = []
        for texts, topic in zip(clusters, topics):
            summary = generate_llm_summary(texts, topic, client=warm_client)
            results.append(summary)
        elapsed = time.perf_counter() - start

        successes = sum(1 for r in results if r)
        print(f"\n  batch 5 summaries: {elapsed:.2f}s total, "
              f"{elapsed/5:.2f}s/summary, {successes}/5 succeeded")


# ═══════════════════════════════════════════════════════════════════════════
# 3. PROMPT CONSTRUCTION (no LLM call)
# ═══════════════════════════════════════════════════════════════════════════

class TestPromptConstruction:
    """Benchmark prompt building — the CPU overhead before inference."""

    def test_build_excerpts_5(self, benchmark, cluster_texts_5):
        """Build excerpt text for 5-chunk cluster."""
        benchmark(_build_cluster_excerpts, cluster_texts_5)

    def test_build_excerpts_10(self, benchmark, cluster_texts_10):
        """Build excerpt text for 10-chunk cluster."""
        benchmark(_build_cluster_excerpts, cluster_texts_10)

    def test_build_full_prompt_5(self, benchmark, cluster_texts_5):
        """Build complete prompt including template formatting."""

        def build():
            excerpts = _build_cluster_excerpts(cluster_texts_5)
            return CLUSTER_SUMMARY_PROMPT.format(
                topic="flock locking", excerpts=excerpts,
            )

        benchmark(build)

    def test_build_consolidation_prompt(self, benchmark):
        """Build consolidation prompt from journal entries."""
        from synapt.recall.consolidate import _build_consolidation_prompt

        entries = _make_journal_entries(5)
        benchmark(
            _build_consolidation_prompt,
            entries, [], None,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. CONSOLIDATION INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

class TestConsolidationInference:
    """Benchmark knowledge extraction from journal entries."""

    def test_consolidation_single_cluster(self, benchmark, warm_client):
        """Single consolidation LLM call (1 cluster of 3 entries)."""
        from synapt.recall.consolidate import _build_consolidation_prompt

        entries = _make_journal_entries(3)
        prompt = _build_consolidation_prompt(entries, [], None)

        def consolidate_one():
            return warm_client.chat(
                model=DEFAULT_MODEL,
                messages=[Message(role="user", content=prompt)],
                temperature=0.1,
            )

        benchmark.pedantic(consolidate_one, rounds=1, warmup_rounds=0)

    def test_consolidation_batch_3_clusters(self, warm_client, capsys):
        """Batch of 3 consolidation clusters (realistic session count)."""
        from synapt.recall.consolidate import _build_consolidation_prompt

        clusters = [
            _make_journal_entries(3),
            _make_journal_entries(4),
            _make_journal_entries(5),
        ]

        start = time.perf_counter()
        for cluster in clusters:
            prompt = _build_consolidation_prompt(cluster, [], None)
            warm_client.chat(
                model=DEFAULT_MODEL,
                messages=[Message(role="user", content=prompt)],
                temperature=0.1,
            )
        elapsed = time.perf_counter() - start

        print(f"\n  consolidation batch (3 clusters): {elapsed:.2f}s total, "
              f"{elapsed/3:.2f}s/cluster")


# ═══════════════════════════════════════════════════════════════════════════
# 5. MEMORY WITH MODEL LOADED
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMMemory:
    """Measure peak memory during LLM operations."""

    def test_model_memory_footprint(self, capsys):
        """Memory consumed by loading the 3B model."""
        MLXClient._BASE_CACHE.clear()
        MLXClient._CURRENT_ADAPTER.clear()

        tracemalloc.start()
        tracemalloc.reset_peak()

        client = MLXClient(MLXOptions(max_tokens=MAX_SUMMARY_TOKENS))
        client.chat(
            model=DEFAULT_MODEL,
            messages=[Message(role="user", content="test")],
            temperature=0.1,
        )

        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        print(f"\n  model load + inference peak: {peak_mb:.0f}MB")
        # 3B 4-bit model should fit under 3GB
        assert peak_mb < 3000, f"Model memory {peak_mb:.0f}MB exceeds 3GB"

    def test_inference_incremental_memory(self, warm_client, capsys):
        """Incremental memory per inference call (model already loaded)."""
        texts = _make_cluster_texts(5)

        tracemalloc.start()
        tracemalloc.reset_peak()

        for _ in range(3):
            generate_llm_summary(texts, "flock locking", client=warm_client)

        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        print(f"\n  3x inference incremental peak: {peak_mb:.1f}MB")
        # Incremental inference should be well under 500MB
        assert peak_mb < 500, f"Inference memory {peak_mb:.0f}MB exceeds 500MB"


# ═══════════════════════════════════════════════════════════════════════════
# 6. QUALITY GATE OVERHEAD
# ═══════════════════════════════════════════════════════════════════════════

class TestQualityGate:
    """Benchmark the quality gate that rejects hallucinated summaries."""

    def test_quality_gate_accept(self, benchmark, warm_client, cluster_texts_5):
        """Full pipeline including quality gate (expected: accept)."""
        benchmark.pedantic(
            generate_llm_summary,
            args=(cluster_texts_5, "flock locking"),
            kwargs={"client": warm_client},
            rounds=1, warmup_rounds=0,
        )

    def test_quality_gate_short_input(self, benchmark, warm_client):
        """Quality gate with minimal input (2 short chunks)."""
        texts = [
            {"user_text": "fix flock", "assistant_text": "Fixed."},
            {"user_text": "test flock", "assistant_text": "Tests pass."},
        ]
        benchmark.pedantic(
            generate_llm_summary,
            args=(texts, "flock"),
            kwargs={"client": warm_client},
            rounds=1, warmup_rounds=0,
        )
