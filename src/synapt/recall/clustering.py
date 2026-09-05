"""Topic clustering for recall chunks using token-overlap similarity.

Groups related chunks into clusters using greedy union-based Jaccard
clustering. No LLM or embeddings required — works purely on token overlap,
which is highly discriminative for code/engineering text where distinctive
tokens like "flock", "TOCTOU", "compact" carry strong signal.

Phase 1 of the intermediate storage layer.
Threshold raised from 0.15 to 0.20 for 85% cluster stability.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re as _re
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from synapt.recall.bm25 import _tokenize

if TYPE_CHECKING:
    from synapt.recall.core import TranscriptChunk
    from synapt.recall.storage import RecallDB

logger = logging.getLogger(__name__)

# Message type for LLM summaries
from synapt._models.base import Message
from synapt.recall._model_router import DEFAULT_DECODER_MODEL as DEFAULT_MODEL

# Tokens that appear in nearly every chunk and carry no topic signal.
# Kept minimal — the stemmer already normalizes inflections.
_STOP_TOKENS = frozenset({
    "the", "is", "it", "to", "in", "for", "of", "and", "or", "an",
    "that", "this", "with", "on", "at", "by", "from", "as", "be",
    "was", "were", "are", "been", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "can", "may", "not",
    "but", "if", "then", "so", "we", "you", "use", "using", "used",
    "let", "me", "here", "file", "code", "run", "add", "fix", "make",
    "look", "need", "now", "get", "set", "new", "also", "just",
    # Common in Claude Code transcripts but carry no topic signal
    "sure", "yes", "okay", "right", "thank", "good", "like", "want",
    "know", "think", "try", "see", "check", "work", "change", "update",
    "call", "test", "return", "function", "method", "class", "import",
    "line", "error", "output", "result", "value", "type", "name",
    "path", "string", "list", "dict", "true", "false", "none",
    "arg", "param", "var", "data", "config", "option", "default",
    "first", "each", "other", "more", "than", "about", "when",
    "them", "they", "their", "your", "our", "its", "which", "what",
    "how", "being", "some", "all", "any", "only", "most",
    "into", "after", "before", "between", "over", "under", "same",
    "command", "message", "otherwise", "caveat",
    "pass", "fail", "read", "write", "open", "close", "move",
    "show", "print", "found", "create", "exist",
    "current", "already", "still", "actually", "instead",
})

# Clustering parameters
JACCARD_THRESHOLD = 0.20  # Raised from 0.15 for tighter clusters
MIN_CLUSTER_SIZE = 2      # Singleton clusters are not useful
MAX_CLUSTER_SIZE = 20     # Split threshold (future)
MIN_TOKENS = 3            # Chunks with fewer tokens are noise


def _chunk_tokens(chunk: TranscriptChunk) -> set[str]:
    """Extract distinctive tokens from a chunk's text content."""
    text = f"{chunk.user_text} {chunk.assistant_text}"
    tokens = _tokenize(text)
    return {t for t in tokens if t not in _STOP_TOKENS and len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def _cluster_id(chunk_ids: list[str]) -> str:
    """Deterministic cluster ID from sorted chunk IDs.

    Format: "clust-<first 12 hex chars of SHA1>".
    """
    key = "\n".join(sorted(chunk_ids))
    sha = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"clust-{sha}"


def _extract_topic(
    token_sets: list[set[str]],
    global_df: Counter[str],
    n_docs: int,
) -> str:
    """Extract a topic label from cluster tokens using DF-IDF keywords.

    Args:
        token_sets: Token sets for chunks in this cluster.
        global_df: Precomputed document frequency across ALL chunks.
        n_docs: Total number of chunks (for IDF denominator).
    """
    if n_docs == 0:
        return "unknown"

    # Cluster document frequency: how many cluster members contain each token
    cluster_df: Counter[str] = Counter()
    for ts in token_sets:
        cluster_df.update(ts)

    # DF-IDF score: tokens frequent in this cluster but rare globally
    # Allow code identifiers (with underscores/dots) alongside pure words
    scores: dict[str, float] = {}
    for token, count in cluster_df.items():
        if len(token) < 4:
            continue
        # Accept pure words and code identifiers (e.g., recall_build, bm25.py)
        if not (token.isalpha() or "_" in token or "." in token):
            continue
        doc_freq = global_df.get(token, 1)
        idf = math.log(n_docs / doc_freq) if doc_freq < n_docs else 0.1
        scores[token] = count * idf

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_tokens = [tok for tok, _ in ranked[:3]]
    if not top_tokens:
        return "unknown"
    return " ".join(top_tokens)


def cluster_chunks(
    chunks: list[TranscriptChunk],
    threshold: float = JACCARD_THRESHOLD,
) -> list[dict]:
    """Group chunks by token overlap using greedy Jaccard clustering.

    For each chunk (newest-first), finds the best-matching cluster by
    Jaccard similarity against the cluster's token union. If similarity
    >= threshold, adds to that cluster. Otherwise creates a new cluster.

    Stability comes from the union's "snowball" property: once a token
    enters a cluster's signature, it stays. Combined with the raised
    threshold (0.20 vs 0.15), this prevents over-absorption while
    maintaining 85% cluster ID stability across rebuilds.

    Returns a list of cluster dicts ready for storage, each containing:
        - cluster_id: deterministic ID from member chunk IDs
        - topic: TF-IDF extracted topic label
        - chunk_ids: list of member chunk IDs
        - session_ids: unique session IDs in the cluster
        - date_start: earliest chunk timestamp
        - date_end: latest chunk timestamp
        - chunk_count: number of member chunks
    """
    if not chunks:
        return []

    # Compute token sets for all chunks
    chunk_token_map: dict[int, set[str]] = {}
    for i, chunk in enumerate(chunks):
        tokens = _chunk_tokens(chunk)
        if len(tokens) >= MIN_TOKENS:
            chunk_token_map[i] = tokens

    # Sort by timestamp descending (newest first) for recency bias
    sorted_indices = sorted(
        chunk_token_map.keys(),
        key=lambda i: chunks[i].timestamp,
        reverse=True,
    )

    # Greedy union-based clustering with inverted index for candidate pruning.
    # Instead of comparing each chunk against ALL clusters (O(n*C)),
    # we maintain token → cluster_indices and count per-cluster overlap.
    # Only compute full Jaccard for clusters exceeding the minimum overlap
    # needed to possibly reach the threshold.
    #
    # Math: Jaccard(A,B) >= t requires |A∩B| >= t*(|A|+|B|)/(1+t).
    # Counting shared tokens via the inverted index is O(|chunk_tokens|),
    # much cheaper than computing Jaccard for every cluster.
    clusters_wip: list[tuple[set[str], list[int]]] = []
    # Inverted index: token → list of cluster indices that contain it
    token_to_clusters: dict[str, list[int]] = {}
    # Precompute threshold factor: t/(1+t)
    threshold_factor = threshold / (1.0 + threshold)

    for idx in sorted_indices:
        chunk_tokens = chunk_token_map[idx]
        chunk_size = len(chunk_tokens)
        best_score = 0.0
        best_cluster_idx = -1

        # Count shared tokens per candidate cluster via inverted index
        overlap_counts: dict[int, int] = {}
        for token in chunk_tokens:
            if token in token_to_clusters:
                for ci in token_to_clusters[token]:
                    overlap_counts[ci] = overlap_counts.get(ci, 0) + 1

        # Only compute full Jaccard where minimum overlap is met
        for ci, overlap in overlap_counts.items():
            cluster_tokens, _members = clusters_wip[ci]
            min_overlap = threshold_factor * (chunk_size + len(cluster_tokens))
            if overlap < min_overlap:
                continue
            score = _jaccard(chunk_tokens, cluster_tokens)
            if score > best_score:
                best_score = score
                best_cluster_idx = ci

        if best_score >= threshold and best_cluster_idx >= 0:
            ct, members = clusters_wip[best_cluster_idx]
            if len(members) < MAX_CLUSTER_SIZE:
                new_tokens = chunk_tokens - ct
                ct.update(chunk_tokens)
                members.append(idx)
                for token in new_tokens:
                    token_to_clusters.setdefault(token, []).append(best_cluster_idx)
            else:
                new_ci = len(clusters_wip)
                clusters_wip.append((set(chunk_tokens), [idx]))
                for token in chunk_tokens:
                    token_to_clusters.setdefault(token, []).append(new_ci)
        else:
            new_ci = len(clusters_wip)
            clusters_wip.append((set(chunk_tokens), [idx]))
            for token in chunk_tokens:
                token_to_clusters.setdefault(token, []).append(new_ci)

    # Precompute global document frequency ONCE for topic extraction
    n_docs = len(chunk_token_map)
    global_df: Counter[str] = Counter()
    for ts in chunk_token_map.values():
        global_df.update(ts)

    # Convert to output format, filtering singletons and empties
    now = datetime.now(timezone.utc).isoformat()
    result: list[dict] = []

    for _cluster_tokens, member_indices in clusters_wip:
        if len(member_indices) < MIN_CLUSTER_SIZE:
            continue

        member_chunks = [chunks[i] for i in member_indices]
        member_token_sets = [chunk_token_map[i] for i in member_indices]
        chunk_ids = [c.id for c in member_chunks]
        session_ids = sorted(set(c.session_id for c in member_chunks))
        timestamps = [c.timestamp for c in member_chunks if c.timestamp]

        topic = _extract_topic(member_token_sets, global_df, n_docs)

        result.append({
            "cluster_id": _cluster_id(chunk_ids),
            "topic": topic,
            "cluster_type": "topic",
            "session_ids": session_ids,
            "branch": None,
            "date_start": min(timestamps) if timestamps else None,
            "date_end": max(timestamps) if timestamps else None,
            "chunk_count": len(chunk_ids),
            "chunk_ids": chunk_ids,
            "status": "active",
            "created_at": now,
            "updated_at": now,
        })

    # Sort by date_end descending (most recent cluster first)
    result.sort(key=lambda c: c.get("date_end") or "", reverse=True)
    return result


def _content_hash(chunk_texts: list[dict]) -> str:
    """Compute a stable content hash for a cluster's chunk texts.

    Uses SHA-256 over sorted, normalized user+assistant text. Two clusters
    with the same chunk content produce the same hash, even if cluster IDs
    differ due to membership ordering changes.
    """
    parts: list[str] = []
    sorted_texts = sorted(
        chunk_texts,
        key=lambda c: (c.get("user_text", ""), c.get("assistant_text", "")),
    )
    for ct in sorted_texts:
        user = (ct.get("user_text") or "").strip()
        asst = (ct.get("assistant_text") or "").strip()
        parts.append(f"{user}\x00{asst}")
    blob = "\x01".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def generate_concat_summary(
    chunks: list[TranscriptChunk],
    max_tokens: int = 200,
) -> str:
    """Generate a cheap summary by concatenating assistant text snippets.

    Takes the first 2 chunks (by timestamp) and concatenates their
    assistant_text, truncated to max_tokens. This is the zero-cost
    fallback when no LLM is available.
    """
    if not chunks or max_tokens <= 0:
        return ""

    # Sort chronologically for narrative coherence
    sorted_chunks = sorted(chunks, key=lambda c: c.timestamp)

    parts: list[str] = []
    token_budget = max_tokens
    for chunk in sorted_chunks[:2]:
        if token_budget <= 0:
            break
        text = chunk.assistant_text.strip()
        if not text:
            continue
        # Rough token estimate: 1 token ~= 4 chars
        available_chars = token_budget * 4
        if len(text) > available_chars:
            text = text[:available_chars] + "..."
        parts.append(text)
        token_budget -= len(text) // 4

    return " ".join(parts)


# ---------------------------------------------------------------------------
# LLM cluster summary generation (Phase 9)
# ---------------------------------------------------------------------------

MAX_CHUNK_CHARS = 4000  # ~1K tokens — fits in 3B context budget
MAX_SUMMARY_TOKENS = 300  # LLM output cap

CLUSTER_SUMMARY_PROMPT = """\
You are summarizing a cluster of related programming session excerpts. \
The cluster covers the topic: {topic}

Produce a concise 2-4 sentence summary that captures:
- What was worked on (specific files, features, bugs)
- Key decisions or outcomes
- Current state (what works, what's pending)

Be specific and concrete. Use technical terms. Do not repeat the topic label.
Output ONLY the summary text, no headings or bullet points.

Excerpts:
{excerpts}

Summary:"""

# FLAN-style prompt for encoder-decoder models (T5 family).
# T5 produces better output with simple, direct instructions rather than
# the detailed role-play prompt used for decoder-only models.
CLUSTER_SUMMARY_PROMPT_T5 = """\
Below are excerpts from a programming session about {topic}. \
Write a 2-4 sentence technical summary covering what was done, \
key decisions, and outcomes.

{excerpts}"""


def _build_cluster_excerpts(
    chunk_texts: list[dict],
    max_chars: int = MAX_CHUNK_CHARS,
) -> str:
    """Build a condensed text block from cluster chunk texts for the LLM."""
    parts: list[str] = []
    char_budget = max_chars
    for i, ct in enumerate(chunk_texts):
        if char_budget <= 0:
            break
        user = (ct.get("user_text") or "").strip()
        asst = (ct.get("assistant_text") or "").strip()
        if not user and not asst:
            continue
        excerpt = f"[{i + 1}]"
        if user:
            excerpt += f" User: {user[:300]}"
        if asst:
            excerpt += f" Assistant: {asst[:500]}"
        parts.append(excerpt)
        char_budget -= len(excerpt)

    return "\n".join(parts)


def create_summary_client() -> object | None:
    """Create a model client for cluster summaries.

    Routes to the best available backend via the model router:
    encoder-decoder (T5) preferred, decoder-only (MLX) as fallback.
    Returns None if no backend is available.
    """
    from synapt.recall._model_router import get_client, RecallTask
    return get_client(RecallTask.SUMMARIZE, max_tokens=MAX_SUMMARY_TOKENS)


# Pattern for CamelCase identifiers (class names, handler names, etc.)
# Matches two+ capitalized words joined together: FarewellHandler, ConnectionPool
_CAMELCASE_RE = _re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")
# Well-known CamelCase names that should not be flagged as hallucinations.
# Only includes names actually matched by _CAMELCASE_RE (two+ Cap segments).
_COMMON_CAMELCASE = frozenset({
    "WebSocket", "IntelliJ", "PyCharm", "JetBrains", "PyTorch",
    "TensorFlow", "HuggingFace", "DataFrame", "DataLoader",
    "GitLab", "BitBucket", "CloudRun", "StackOverflow",
    "OpenAI", "LangChain", "NumPy", "SciPy",
})


def _novel_entities(summary: str, source_text_lower: str) -> set[str]:
    """Find CamelCase entities in summary that don't appear in source text.

    Used as a hallucination detector: if the LLM invents class names,
    handler names, or component names not present in the excerpts,
    the summary is likely fabricated.

    Args:
        summary: The LLM-generated summary text.
        source_text_lower: The source excerpts text, lowercased.

    Returns:
        Set of novel CamelCase entities found in summary but not source.
    """
    entities = set(_CAMELCASE_RE.findall(summary))
    entities -= _COMMON_CAMELCASE
    novel = set()
    for entity in entities:
        if entity.lower() not in source_text_lower:
            novel.add(entity)
    return novel


def generate_llm_summary(
    chunk_texts: list[dict],
    topic: str,
    model: str = "",
    client: object | None = None,
) -> str | None:
    """Generate an LLM-powered cluster summary.

    Args:
        chunk_texts: List of dicts with user_text/assistant_text keys.
        topic: The cluster topic label (from DF-IDF extraction).
        model: Model to use. Auto-selected based on client type if empty.
        client: Reusable model client. Created via router if not provided.

    Returns the summary string, or None if no backend is available or
    inference fails.
    """
    if not chunk_texts:
        return None

    if client is None:
        client = create_summary_client()
    if client is None:
        return None

    excerpts = _build_cluster_excerpts(chunk_texts)
    if not excerpts:
        return None

    # Select model and prompt based on client architecture
    from synapt.recall._model_router import is_encoder_decoder, get_encoder_decoder_model
    if is_encoder_decoder(client):
        model = model or get_encoder_decoder_model()
        prompt = CLUSTER_SUMMARY_PROMPT_T5.format(topic=topic, excerpts=excerpts)
    else:
        model = model or DEFAULT_MODEL
        prompt = CLUSTER_SUMMARY_PROMPT.format(topic=topic, excerpts=excerpts)

    try:
        response = client.chat(
            model=model,
            messages=[Message(role="user", content=prompt)],
            temperature=0.1,
        )
    except Exception as exc:
        logger.warning("Inference failed for cluster summary: %s", exc)
        return None

    summary = response.strip()
    if not summary:
        return None

    # Quality gate 1: reject if longer than the input (hallucination signal)
    input_len = sum(
        len(ct.get("user_text", "")) + len(ct.get("assistant_text", ""))
        for ct in chunk_texts
    )
    if len(summary) > input_len and input_len > 0:
        logger.warning(
            "LLM summary (%d chars) longer than input (%d chars), rejecting",
            len(summary), input_len,
        )
        return None

    # Quality gate 2: reject if summary introduces entities not in source
    source_text = excerpts.lower()
    novel = _novel_entities(summary, source_text)
    if len(novel) >= 2:
        logger.warning(
            "LLM summary has %d novel entities not in source (%s), rejecting",
            len(novel), ", ".join(sorted(novel)[:5]),
        )
        return None

    return summary


MIN_CONTENT_RATIO = 0.3    # At least 30% of chunks must have assistant text
MIN_AVG_CONTENT_LEN = 30   # Average assistant text must be at least 30 chars
MIN_DIVERSITY_RATIO = 0.3  # At least 30% of chunks must have unique content


def _has_meaningful_content(chunk_texts: list[dict]) -> bool:
    """Check if a cluster has enough real content to justify an LLM summary.

    Three-level filter:
    1. Content presence: enough chunks must have non-empty assistant text
    2. Content length: average assistant text must be substantial
    3. Content diversity: must have diverse responses (not repetitive noise)

    Noise clusters (tool interrupts, stdout, stale notifications, API errors)
    are the LARGEST clusters because identical system messages repeat, cluster
    strongly, and hit MAX_CLUSTER_SIZE. They pass content length checks
    (API errors are long) but fail diversity (same message repeated 20x).
    Meaningful programming discussion is inherently diverse.
    """
    if not chunk_texts:
        return False

    non_empty = 0
    total_len = 0
    # Track unique text prefixes for diversity check
    prefixes: set[str] = set()

    for ct in chunk_texts:
        asst = (ct.get("assistant_text") or "").strip()
        if len(asst) > 20:
            non_empty += 1
            total_len += len(asst)
            # First 100 chars capture enough to distinguish unique responses
            prefixes.add(asst[:100])

    n = len(chunk_texts)
    ratio = non_empty / n
    avg_len = total_len / n
    diversity = len(prefixes) / n if n > 0 else 0

    return (
        ratio >= MIN_CONTENT_RATIO
        and avg_len >= MIN_AVG_CONTENT_LEN
        and diversity >= MIN_DIVERSITY_RATIO
    )


def upgrade_large_cluster_summaries(
    db: "RecallDB",
    min_chunks: int = 5,
    max_upgrades: int = 5,
) -> int:
    """Generate LLM summaries for the largest clusters that lack them.

    Targets clusters by size (chunk_count) rather than access frequency,
    ensuring the most information-rich clusters get quality summaries
    regardless of how often they've been searched.

    Args:
        db: RecallDB instance.
        min_chunks: Minimum chunk count to qualify for LLM summary.
        max_upgrades: Maximum number of LLM summaries to generate per call.

    Returns the number of summaries generated.
    """
    # Check if any LLM backend is available
    test_client = create_summary_client()
    if test_client is None:
        return 0

    # Find large clusters without LLM summaries, ordered by size descending
    rows = db._conn.execute(
        "SELECT c.cluster_id, c.topic, c.chunk_count "
        "FROM clusters c "
        "LEFT JOIN cluster_summaries cs "
        "  ON c.cluster_id = cs.cluster_id AND cs.method = 'llm' "
        "WHERE c.status = 'active' "
        "  AND c.chunk_count >= ? "
        "  AND cs.cluster_id IS NULL "
        "ORDER BY c.chunk_count DESC",
        (min_chunks,),
    ).fetchall()

    if not rows:
        return 0

    client = test_client
    count = 0

    for row in rows:
        cluster_id = row[0]
        topic = row[1]
        chunk_texts = db.get_cluster_chunk_texts(cluster_id)
        if not chunk_texts:
            continue

        # Skip noise clusters: system artifacts (tool use, stdout, interrupts)
        # cluster into the largest groups because they're highly repetitive
        # but have zero informational value. Detect by checking assistant_text.
        if not _has_meaningful_content(chunk_texts):
            logger.debug(
                "Skipping noise cluster %s (%d chunks, topic: %s)",
                cluster_id, row[2], topic,
            )
            continue

        # Check content hash — reuse existing summary if content unchanged
        chash = _content_hash(chunk_texts)
        existing = db.find_summary_by_content_hash(chash)
        if existing:
            db.save_cluster_summary(
                cluster_id, existing, method="llm", content_hash=chash,
            )
            count += 1
            logger.debug(
                "Reused LLM summary for cluster %s via content hash",
                cluster_id,
            )
        else:
            summary = generate_llm_summary(chunk_texts, topic, client=client)
            if summary:
                db.save_cluster_summary(
                    cluster_id, summary, method="llm", content_hash=chash,
                )
                count += 1
                logger.debug(
                    "LLM summary generated for large cluster %s (%d chunks)",
                    cluster_id, row[2],
                )

        if count >= max_upgrades:
            break

    # Clean up orphaned LLM summaries that were kept for content hash matching.
    # Now that upgrade is complete, any orphans whose content_hash was reused
    # above (or not needed) can be safely removed.
    db._conn.execute(
        "DELETE FROM cluster_summaries "
        "WHERE method = 'llm' "
        "AND cluster_id NOT IN (SELECT cluster_id FROM clusters)"
    )
    db._conn.commit()

    return count


# recall#435 follow-on: bounded reclustering for chunks a skipped build left
# stale. Defaults are first estimates, not yet validated against a real
# per-chunk memory measurement -- see the PR that lands this comment for the
# measured number and peak memory_pressure before it becomes the shipped
# default.
DEFAULT_RECLUSTER_BATCH = 2000
DEFAULT_RECLUSTER_REFUSE_ABOVE = 50_000


def stale_transcript_chunk_ids(db: "RecallDB") -> list[str]:
    """Transcript chunk ids (turn_index >= 0) with no topic-cluster membership.

    recall#435's ``skip_clustering`` build path saves chunks to FTS5 without
    clustering them, which makes storage.py's own long-standing assumption
    ("clustering is fully rebuilt on every recall build") false. This is
    what makes that assumption checkable again.

    Cluster membership always lives in the metadata/index shard -- both
    ``save_clusters`` and ``append_clusters`` route there on
    ``ShardedRecallDB`` -- while chunks may be split across per-shard data
    DBs. Reads chunk HEADERS only (id + turn_index, no chunk text), across
    every shard: cheap and O(chunk count), not the full-text O(store size)
    cost that made full-corpus clustering itself OOM.
    """
    clustered_ids = {
        row[0]
        for row in db._conn.execute(
            "SELECT DISTINCT cc.chunk_id FROM cluster_chunks cc "
            "JOIN clusters cl ON cl.cluster_id = cc.cluster_id "
            "WHERE cl.cluster_type = 'topic'"
        ).fetchall()
    }
    headers = db.load_chunk_headers()
    return [c.id for c in headers if c.turn_index >= 0 and c.id not in clustered_ids]


def _select_recluster_batch(
    db: "RecallDB", stale_ids: list[str], batch_size: int
) -> tuple[list[str], int, int]:
    """never-attempted stale chunks first, oldest first; fall back
    to already-attempted ones only once the fresh queue is exhausted.

    Measured on the real store: without this preference, 98.9%
    of a batch overlaps the previous run's batch when most of it fails to
    cluster, so the op makes almost no forward progress. Returns
    ``(batch_ids, fresh_count, fallback_count)``.
    """
    attempted = db.get_recluster_attempted_ids()
    fresh = [cid for cid in stale_ids if cid not in attempted]
    fallback_pool = [cid for cid in stale_ids if cid in attempted]

    batch = fresh[:batch_size]
    fresh_count = len(batch)
    if fresh_count < batch_size:
        batch = batch + fallback_pool[: batch_size - fresh_count]
    return batch, fresh_count, len(batch) - fresh_count


def recluster_stale_chunks(
    db: "RecallDB",
    batch_size: int = DEFAULT_RECLUSTER_BATCH,
    refuse_above: int = DEFAULT_RECLUSTER_REFUSE_ABOVE,
) -> dict:
    """Cluster a bounded batch of stale chunks, preferring never-tried ones
    first; report the backlog.

    The memory bound that actually prevents a repeat of recall#435's OOM:
    this NEVER reloads the already-clustered corpus. It clusters only the
    selected batch among itself and calls ``append_clusters`` (additive), so
    a stale chunk that is Jaccard-similar to an EXISTING cluster still gets
    its own NEW cluster rather than being merged into the old one's
    token-union. Named tradeoff, not a silent gap -- merging into existing
    clusters is a later, separate lane: the real store clears only ~1-4%
    of a batch per run, so most of the backlog cannot cluster with itself
    at all and needs the merge lane, not a bigger batch or a smarter skip.

    Batch SELECTION prefers chunks ``recluster_attempts`` has never seen:
    a chunk that fails to cluster is marked attempted and routed around on
    future runs until every never-tried chunk is exhausted, which is what
    keeps the op making forward progress instead of reselecting the same
    stuck chunks every time.

    Above ``refuse_above`` stale chunks (total backlog, not batch
    composition), refuses outright and reports the backlog plus the command
    that would drain it in batches, rather than attempting one huge batch:
    that many stale chunks means something is wrong (``skip_clustering``
    left on for a long stretch, most likely), not that a bigger single
    batch is safe.
    """
    stale_ids = stale_transcript_chunk_ids(db)
    total_stale = len(stale_ids)

    def _drain_command(remaining: int) -> str:
        if remaining <= 0:
            return ""
        # Ceil division is a LOWER BOUND on runs needed, not a prediction:
        # it assumes every chunk in every future batch actually clusters,
        # which cluster_chunks()'s own MIN_CLUSTER_SIZE filtering does not
        # guarantee (measured on the real store: 71 of 2000 selected chunks,
        # 3.6%, actually formed a saved cluster in one run -- the rest stay
        # stale and get reselected). "At least" is the honest word; a flat
        # "N more runs" would overpromise convergence this function cannot
        # itself guarantee.
        runs = -(-remaining // batch_size)  # ceil division
        return (
            f"synapt maintain --recluster --recluster-batch {batch_size} "
            f"(at least {runs} more run{'s' if runs != 1 else ''} to drain the backlog)"
        )

    if total_stale > refuse_above:
        return {
            "refused": True,
            "total_stale_at_start": total_stale,
            "batches_run": 0,
            "chunks_clustered": 0,
            "still_stale": total_stale,
            "fresh_in_batch": 0,
            "fallback_in_batch": 0,
            "drain_command": _drain_command(total_stale),
        }

    batch_ids, fresh_count, fallback_count = _select_recluster_batch(db, stale_ids, batch_size)
    if not batch_ids:
        return {
            "refused": False,
            "total_stale_at_start": 0,
            "batches_run": 0,
            "chunks_clustered": 0,
            "still_stale": 0,
            "fresh_in_batch": 0,
            "fallback_in_batch": 0,
            "drain_command": "",
        }

    id_rowid_map = db.get_chunk_id_rowid_map()
    rowids = [id_rowid_map[cid] for cid in batch_ids if cid in id_rowid_map]
    chunk_map = db.load_chunks_by_rowids(rowids)
    batch_chunks = [chunk_map[r] for r in rowids if r in chunk_map]

    clusters = cluster_chunks(batch_chunks)
    if clusters:
        memberships = [
            (cl["cluster_id"], cid, cl["created_at"])
            for cl in clusters
            for cid in cl["chunk_ids"]
        ]
        db.append_clusters(clusters, memberships)

    still_stale_ids = set(stale_transcript_chunk_ids(db))
    # Mark only batch members that FAILED to cluster this run -- a chunk
    # that succeeded is no longer stale and will never be reselected, so it
    # needs no attempt row (see the table's own comment in storage.py).
    failed_this_run = [cid for cid in batch_ids if cid in still_stale_ids]
    if failed_this_run:
        db.mark_recluster_attempted(failed_this_run, run_id=uuid.uuid4().hex[:12])

    still_stale = len(still_stale_ids)
    return {
        "refused": False,
        "total_stale_at_start": total_stale,
        "batches_run": 1,
        "chunks_clustered": total_stale - still_stale,
        "still_stale": still_stale,
        "fresh_in_batch": fresh_count,
        "fallback_in_batch": fallback_count,
        "drain_command": _drain_command(still_stale),
    }
