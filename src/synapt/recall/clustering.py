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

# A stale-chunk-to-existing-cluster match compares against a
# bounded, persisted per-cluster signature (top N tokens by within-cluster
# document frequency), not the full member set. Starting values, subject to
# revision after a hand-read of real matches -- not tuned from a synthetic
# target.
TOP_SIGNATURE_TOKENS = 64        # signature size cap
CONTAINMENT_THRESHOLD = 0.20     # fraction of the signature that must be present
MIN_SHARED_SIGNATURE_TOKENS = 8  # absolute floor: ratio alone lets a tiny
                                  # signature "match" on a handful of coincidental tokens

# Harness-injected wrapper blocks, verified with matching
# open/close counts on a real 2000-chunk stale batch (skill 162/162,
# command-message 111/111, local-command-stdout 9/9). These are structural
# artifacts of HOW a message reached the transcript, not conversational
# content, and the SAME block repeats verbatim across many otherwise-
# unrelated turns (measured: 477 of 2000 stale chunks in that batch carried
# a <skill> block alone). Stripped unconditionally, before any statistical
# guard -- exact and cheap beats probabilistic where the shape is known.
# ``task-notification`` is NOT here: it never closes in this corpus (0/35
# on the same batch), evidently truncated upstream before the closing tag
# -- left to the per-batch data-derived guard in recluster_stale_chunks,
# which does not need a clean tag pair to work.
_HARNESS_PREAMBLE_TAGS = ("skill", "command-message", "local-command-stdout")
_HARNESS_PREAMBLE_RE = _re.compile(
    r"<(" + "|".join(_HARNESS_PREAMBLE_TAGS) + r")>.*?</\1>", _re.DOTALL,
)


def _strip_harness_preamble(text: str) -> str:
    """Remove KNOWN harness-injected wrapper blocks before tokenizing."""
    return _HARNESS_PREAMBLE_RE.sub(" ", text)


# recall's OWN synthetic restatement, not the harness's. core.py
# gives every sub-chunk PAST THE FIRST of a long assistant reply this string
# as its ENTIRE user_text (see core.py's chunk-splitting loop and resume.py's
# _is_continuation, which checks the same literal prefix): a truncated echo
# of the ONE user turn that started the exchange. Two sub-chunks of the same
# exchange therefore share this exact string verbatim, which inflates their
# apparent similarity for reasons that have nothing to do with what either
# chunk actually SAYS -- measured store-wide: 102,669 of 169,053 chunks carry
# it, and one traced merge (found via hand-read) matched almost entirely on
# a shared echo, not shared substance. Since it is the WHOLE field rather
# than a span inside a larger blob (unlike the tags above), the fix is to
# drop the field entirely when it starts with this marker, not to regex out
# a substring -- the echoed text itself can contain stray punctuation
# (it's an arbitrary 100-char slice of real user text) that would break a
# span-based end-of-block match.
_CONTEXT_ECHO_PREFIX = "(context: User previously asked:"


def _has_letter(token: str) -> bool:
    """True if ``token`` carries at least one alphabetic character.

    Hand-read finding: a "User selected: <path>\\nN\\u2192<code>" editor
    line-selection echo puts bare decimal line numbers into the token
    stream (``_tokenize`` keeps digit-only strings as-is), and two entirely
    unrelated files that both happen to have a selection starting around
    the same line number then share dozens of "distinctive" tokens that are
    really just consecutive integers -- coincidence, not topic overlap.
    Measured: one real cluster signature was 64/64 bare numbers, no
    topical content at all, and would contentedly match any other
    numbered-code excerpt in a similar line range. There is no dash or
    arrow token to special-case: ``_tokenize``'s charset (``[a-zA-Z0-9_.]``)
    already treats ``-`` and ``\\u2192`` as separators, so "100-200" and
    "100\\u2192200" both arrive as plain per-number tokens already, not one
    joined token -- the general rule (a content token has at least one
    letter) is what actually needs to hold, not a shape-specific strip.
    """
    return any(c.isalpha() for c in token)


def _chunk_tokens(
    chunk: TranscriptChunk, extra_stopwords: frozenset[str] = frozenset(),
) -> set[str]:
    """Extract distinctive tokens from a chunk's text content.

    Drops recall's own synthetic context-echo user_text FIRST (see
    ``_CONTEXT_ECHO_PREFIX`` -- it is not the turn's content, so a
    comparison that can see it compares the echo, not the chunk), then
    strips known harness preamble blocks (structural, unconditional, see
    ``_HARNESS_PREAMBLE_TAGS``), then applies ``extra_stopwords`` -- a
    DATA-DERIVED, per-caller stoplist (see ``compute_boilerplate_stoplist``)
    for whatever boilerplate the two structural strips above do not have
    an exact shape for -- and finally drops any token with no letters at
    all (see ``_has_letter``): a run of digits can be pervasive between two
    otherwise-unrelated chunks by pure coincidence in a way an English or
    code word essentially cannot. Default empty: every existing caller is
    unaffected until it opts in.
    """
    user_text = (
        "" if chunk.user_text.lstrip().startswith(_CONTEXT_ECHO_PREFIX)
        else chunk.user_text
    )
    text = _strip_harness_preamble(f"{user_text} {chunk.assistant_text}")
    tokens = _tokenize(text)
    return {
        t for t in tokens
        if t not in _STOP_TOKENS and t not in extra_stopwords and len(t) > 2
        and _has_letter(t)
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def _cluster_signature_tokens(
    token_sets: list[set[str]],
    top_n: int = TOP_SIGNATURE_TOKENS,
    extra_stopwords: frozenset[str] = frozenset(),
) -> list[str]:
    """A cluster's bounded signature: the top ``top_n`` tokens by
    WITHIN-cluster document frequency (how many member chunks contain the
    token -- each ``ts`` is a per-chunk SET, so ``Counter.update(ts)``
    counts exactly one per member that has the token, never raw occurrence
    count). Same quantity ``_extract_topic`` computes for its own DF-IDF
    ranking; kept as a separate small pass here rather than threading a
    second return value through that function's existing call sites.

    ``extra_stopwords`` is filtered here TOO, not only at ``_chunk_tokens``:
    if ``token_sets`` ever arrives from a path that does not route through
    ``_chunk_tokens`` (a call site added later, say), the signature is still
    protected from data-derived boilerplate rather than silently trusting
    an upstream filter that may not apply.
    """
    document_frequency: Counter[str] = Counter()
    for ts in token_sets:
        document_frequency.update(t for t in ts if t not in extra_stopwords)
    return [tok for tok, _ in document_frequency.most_common(top_n)]


def compute_boilerplate_stoplist(
    token_sets: dict[str, set[str]], min_fraction: float = 0.20,
) -> list[tuple[str, float]]:
    """A DATA-DERIVED stoplist: tokens present in more than ``min_fraction``
    of the given token sets, sorted by fraction descending.

    Population-agnostic -- ``token_sets`` can be cluster signatures OR a
    batch of chunks, keyed by whatever id the caller has. The POPULATION
    it is computed over matters more than the mechanism: measured on the
    real store, a skill-preamble token was under 2% of persisted cluster
    signatures (which skew toward older, already-clustered content) but
    23.9% of one actual stale-chunk batch (recently-fired activity), so a
    stoplist derived from signatures did not see the batch's problem at
    all. ``recluster_stale_chunks`` calls this on the CURRENT batch, never
    on cluster signatures -- a signature-derived stoplist was tried and
    rejected: on this corpus its top tokens are the team's own real
    process vocabulary (agent names, "gate", "ratify", "merge"), and
    stripping those would remove real topical signal, not just noise.

    Unlike ``_STOP_TOKENS`` (hand-curated, general-English function words),
    this catches vocabulary that is common for reasons that have nothing to
    do with topical similarity -- concretely, whatever a structural strip
    (see ``_HARNESS_PREAMBLE_TAGS``, ``_CONTEXT_ECHO_PREFIX``) does not have
    an exact shape for. Returns empty on no token sets (division by zero
    avoided, not silently wrong).

    A batch-derived drop list naming this team's OWN process words (e.g.
    "dev", "review", "channel" -- measured on a real batch) is expected and
    is not the signature-rejection problem repeating: it is recomputed from
    THIS batch alone, discarded when the run ends, and never written to a
    persisted signature, so it cannot permanently erase a topic's real
    vocabulary the way stripping it from a signature would.
    """
    total = len(token_sets)
    if total == 0:
        return []
    document_frequency: Counter[str] = Counter()
    for tokens in token_sets.values():
        document_frequency.update(tokens)
    return sorted(
        (
            (tok, count / total)
            for tok, count in document_frequency.items()
            if count / total > min_fraction
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )


def _match_existing_cluster(
    chunk_tokens: set[str],
    cluster_signatures: dict[str, set[str]],
    *,
    min_containment: float = CONTAINMENT_THRESHOLD,
    min_shared_tokens: int = MIN_SHARED_SIGNATURE_TOKENS,
) -> str | None:
    """Best-matching existing cluster id for one chunk's tokens, or None.

    Asymmetric CONTAINMENT of the cluster's signature in the chunk's
    tokens -- ``|signature ∩ chunk| / |signature|`` -- not symmetric
    Jaccard: a signature is capped at ``TOP_SIGNATURE_TOKENS`` while a
    chunk commonly carries far more tokens, so Jaccard's union-sized
    denominator dilutes real overlap toward zero regardless of how
    thematically similar the chunk actually is.

    Containment alone is the false-merge shape for a SMALL signature: a
    10-token signature needs only 3 shared tokens to read 0.30 contained,
    which is coincidence, not similarity. ``min_shared_tokens`` is an
    absolute floor independent of the ratio, so a signature this small
    cannot match without substantial real overlap.

    A signature is also ineligible outright when it holds fewer than
    ``min_shared_tokens`` LETTER-BEARING tokens (see ``_has_letter``),
    checked directly against the signature rather than inferred from
    ``_chunk_tokens`` already excluding non-letter tokens on the chunk
    side: a signature persisted BEFORE that filter existed can still be
    sitting on disk full of coincidental line-number tokens (measured:
    one real signature was 64/64 bare numbers) until its
    next backfill, and this check keeps it from acting as a magnet for
    numbered-code chunks in the meantime, rather than relying only on the
    fact that a freshly-tokenized chunk can no longer contribute matching
    digits to ``shared``.
    """
    best_score = 0.0
    best_id: str | None = None
    for cluster_id, signature in cluster_signatures.items():
        if not signature:
            continue
        if sum(1 for t in signature if _has_letter(t)) < min_shared_tokens:
            continue
        shared = len(signature & chunk_tokens)
        if shared < min_shared_tokens:
            continue
        containment = shared / len(signature)
        if containment >= min_containment and containment > best_score:
            best_score = containment
            best_id = cluster_id
    return best_id


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
    boilerplate_stopwords: frozenset[str] = frozenset(),
) -> list[dict]:
    """Group chunks by token overlap using greedy Jaccard clustering.

    For each chunk (newest-first), finds the best-matching cluster by
    Jaccard similarity against the cluster's token union. If similarity
    >= threshold, adds to that cluster. Otherwise creates a new cluster.

    Stability comes from the union's "snowball" property: once a token
    enters a cluster's signature, it stays. Combined with the raised
    threshold (0.20 vs 0.15), this prevents over-absorption while
    maintaining 85% cluster ID stability across rebuilds.

    ``boilerplate_stopwords`` (see ``compute_boilerplate_stoplist``) is
    filtered out of every chunk's tokens before Jaccard runs: two chunks
    sharing nothing but an identical injected system/skill preamble would
    otherwise read as topically similar on that shared boilerplate alone.
    Default empty -- existing callers are unaffected.

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
        tokens = _chunk_tokens(chunk, boilerplate_stopwords)
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
        signature_tokens = _cluster_signature_tokens(
            member_token_sets, extra_stopwords=boilerplate_stopwords,
        )

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
            "signature_tokens": signature_tokens,
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
DEFAULT_SIGNATURE_BACKFILL_BATCH = 500

# Below this many chunks, "present in >20% of the batch" is not
# a meaningful pervasiveness signal -- at 2 chunks, ONE occurrence is
# already 50%, so the guard would strip the entire batch's vocabulary on
# any shared token at all (caught by the e2e test's small fixture batch).
# The real-store batch this guard was built for is thousands of chunks;
# this floor only matters for small batches, tests included.
MIN_BATCH_FOR_BOILERPLATE_GUARD = 20

# Three rounds of "read the actual merged pairs before trusting
# the aggregate count" each needed an ad-hoc script to reconstruct which
# chunk merged into which cluster, because the receipt only ever carried a
# count. Capping a small sample IN the receipt itself is what makes a
# dry run's hand-read (and any real run's) not require touching the DB
# again after the fact.
MERGE_SAMPLE_SIZE = 10


def backfill_cluster_signatures(
    db: "RecallDB",
    batch_size: int = DEFAULT_SIGNATURE_BACKFILL_BATCH,
    boilerplate_stopwords: frozenset[str] = frozenset(),
) -> dict:
    """Compute and persist a token signature for a bounded batch of clusters
    that do not have one yet -- clusters that predate ``cluster_token_signatures``,
    or one a recent merge invalidated (``merge_chunks_into_cluster`` deletes
    the old signature outright).

    Memory bound: this loads member chunks for at most ``batch_size``
    clusters at a time (each capped at ``MAX_CLUSTER_SIZE``), never every
    cluster's members at once -- the same batching discipline
    ``recluster_stale_chunks`` uses for chunks, applied here to clusters.

    ``boilerplate_stopwords`` (see ``compute_boilerplate_stoplist``) is
    filtered before the signature is computed, same reasoning as
    ``cluster_chunks``. Default empty -- existing callers are unaffected.
    """
    missing = db.active_topic_clusters_missing_signature()
    total_missing = len(missing)
    batch_cluster_ids = missing[:batch_size]
    if not batch_cluster_ids:
        return {"clusters_signed": 0, "clusters_skipped": 0, "clusters_remaining": 0}

    members_by_cluster = db.load_cluster_member_chunk_ids(batch_cluster_ids)
    all_chunk_ids = sorted({cid for ids in members_by_cluster.values() for cid in ids})
    id_rowid_map = db.get_chunk_id_rowid_map()
    rowids = [id_rowid_map[cid] for cid in all_chunk_ids if cid in id_rowid_map]
    chunk_map_by_rowid = db.load_chunks_by_rowids(rowids)
    chunk_by_id = {
        c.id: c
        for r in rowids
        if r in chunk_map_by_rowid
        for c in [chunk_map_by_rowid[r]]
    }

    now = datetime.now(timezone.utc).isoformat()
    signed = 0
    skipped = 0
    for cluster_id in batch_cluster_ids:
        member_ids = members_by_cluster.get(cluster_id, [])
        token_sets = [
            _chunk_tokens(chunk_by_id[cid], boilerplate_stopwords)
            for cid in member_ids
            if cid in chunk_by_id
        ]
        # A cluster with no resolvable member (its rows were deleted out
        # from under it, or member_ids was empty) has nothing to sign from.
        if not token_sets:
            skipped += 1
            continue
        signature = _cluster_signature_tokens(
            token_sets, extra_stopwords=boilerplate_stopwords,
        )
        db.save_cluster_token_signature(cluster_id, signature, now)
        signed += 1

    return {
        "clusters_signed": signed,
        "clusters_skipped": skipped,
        "clusters_remaining": total_missing - signed - skipped,
    }


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
    merge_into_existing: bool = False,
    boilerplate_stopwords: frozenset[str] = frozenset(),
    dry_run: bool = False,
) -> dict:
    """Cluster a bounded batch of stale chunks, preferring never-tried ones
    first; report the backlog.

    The memory bound that actually prevents a repeat of recall#435's OOM:
    this NEVER reloads the already-clustered corpus. It clusters only the
    selected batch among itself and calls ``append_clusters`` (additive).

    When ``merge_into_existing`` is set, a stale chunk gets a first chance
    to join an EXISTING cluster before self-batch clustering ever runs:
    ``load_cluster_token_signatures`` is a cheap read (cluster count, not
    chunk count) of every cluster's PERSISTED bounded signature -- the top
    ``TOP_SIGNATURE_TOKENS`` tokens by within-cluster document frequency,
    not the raw ``search_text`` sample (measured: symmetric Jaccard against
    search_text-derived tokens dilutes to ~0.05-0.10 regardless of real
    thematic overlap, because a chunk's own token set runs far larger than
    a short search_text sample). Each batch chunk is matched by asymmetric
    CONTAINMENT of the signature in its own tokens, with an absolute
    shared-token floor (see ``_match_existing_cluster``). A match grows
    that cluster in place (``merge_chunks_into_cluster`` -- membership
    added, chunk_count and search_text updated, cluster_id UNCHANGED since
    the id may already be referenced elsewhere; the cluster's now-stale
    signature is deleted so it re-enters the backfill queue). Chunks with
    no match fall through to self-batch clustering exactly as before.
    Default False: this changes which chunks a batch clusters, so it stays
    opt-in until the real-store numbers say otherwise. A cluster with no
    persisted signature yet (predates this table, or one that had a signature
    deleted by a recent merge) is simply absent from the comparison --
    ``backfill_cluster_signatures`` is the bounded op that fills it in.

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

    ``dry_run=True`` runs every comparison and computes the SAME receipt
    (``merged_into_existing``, ``chunks_clustered``, ``still_stale`` derived
    from the in-memory decisions rather than a post-write re-query) but
    skips every write (``merge_chunks_into_cluster``, ``append_clusters``,
    ``mark_recluster_attempted``) -- three rounds of hand-
    reading real merges each needed writing to the live store first (or a
    full copy of it) because no dry mode existed; this is that mode. The
    receipt also carries ``merge_samples`` (bounded by ``MERGE_SAMPLE_SIZE``,
    chunk id / target cluster id / target topic) regardless of ``dry_run``,
    so a hand-read never has to reconstruct which chunk went where from a
    before/after diff of the database again.
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
            "merged_into_existing": 0,
            "batch_boilerplate_dropped": [],
            "merge_samples": [],
            "merge_run_id": None,
            "dry_run": dry_run,
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
            "merged_into_existing": 0,
            "batch_boilerplate_dropped": [],
            "merge_samples": [],
            "merge_run_id": None,
            "dry_run": dry_run,
            "drain_command": "",
        }

    id_rowid_map = db.get_chunk_id_rowid_map()
    rowids = [id_rowid_map[cid] for cid in batch_ids if cid in id_rowid_map]
    chunk_map = db.load_chunks_by_rowids(rowids)
    batch_chunks = [chunk_map[r] for r in rowids if r in chunk_map]

    merged_count = 0
    batch_boilerplate_dropped: list[tuple[str, float]] = []
    merge_samples: list[dict] = []
    effective_stopwords = boilerplate_stopwords
    # One run_id per call, stamped on every membership row this
    # run's merges write (see cluster_chunks' schema comment) -- an undo of
    # THIS run's merges is then a query by run_id, never the "shared
    # timestamp" fallback a live-store incident needed before this column
    # existed. Self-batch clustering rows are unaffected (append_clusters
    # does not take a run_id; a full rebuild replaces them wholesale).
    merge_run_id = uuid.uuid4().hex[:12] if merge_into_existing else None
    if merge_into_existing and batch_chunks:
        # Data-derived guard computed from THIS BATCH, not cluster
        # signatures: a stoplist built from signatures (older, already-
        # clustered content) measured under 2% for the exact tokens that
        # were 23.9% of a real stale batch -- signatures and the incoming
        # stale queue are different populations with different composition,
        # so the guard has to look at the population actually being
        # compared. Structural stripping already ran inside _chunk_tokens
        # (see _HARNESS_PREAMBLE_TAGS); this catches whatever that does not
        # have an exact tag for. With the structural strip in place this
        # should be short -- a long list here is itself a finding.
        raw_batch_tokens = {
            c.id: _chunk_tokens(c, boilerplate_stopwords) for c in batch_chunks
        }
        if len(batch_chunks) >= MIN_BATCH_FOR_BOILERPLATE_GUARD:
            batch_boilerplate_dropped = compute_boilerplate_stoplist(
                raw_batch_tokens, min_fraction=0.20,
            )
            effective_stopwords = boilerplate_stopwords | frozenset(
                tok for tok, _frac in batch_boilerplate_dropped
            )

        cluster_token_sets = db.load_cluster_token_signatures()
        now = datetime.now(timezone.utc).isoformat()
        merges_by_cluster: dict[str, list[TranscriptChunk]] = {}
        remaining_chunks: list[TranscriptChunk] = []
        for chunk in batch_chunks:
            tokens = raw_batch_tokens[chunk.id] - effective_stopwords
            # Same noise filter self-batch clustering applies -- a chunk too
            # short to ever form its own cluster is too short to match an
            # existing one meaningfully either.
            target = (
                _match_existing_cluster(tokens, cluster_token_sets)
                if len(tokens) >= MIN_TOKENS
                else None
            )
            if target is not None:
                merges_by_cluster.setdefault(target, []).append(chunk)
            else:
                remaining_chunks.append(chunk)
        # Signatures are a snapshot from the top of this batch: two batch
        # members that both match the SAME existing cluster both merge into
        # it correctly (chunk_count is recomputed, not incremented), but a
        # chunk merged earlier in this loop does not grow the signature a
        # LATER chunk in the same batch is compared against. Named
        # tradeoff for the first working version, not a silent gap.
        for cluster_id, chunks_to_merge in merges_by_cluster.items():
            appended_text = " ".join(
                part
                for c in chunks_to_merge
                for part in (c.user_text, c.assistant_text)
                if part
            )
            if not dry_run:
                db.merge_chunks_into_cluster(
                    cluster_id, [c.id for c in chunks_to_merge], appended_text, now,
                    run_id=merge_run_id,
                )
            merged_count += len(chunks_to_merge)
            for c in chunks_to_merge:
                if len(merge_samples) < MERGE_SAMPLE_SIZE:
                    merge_samples.append({"chunk_id": c.id, "cluster_id": cluster_id})
        batch_chunks = remaining_chunks

    clusters = cluster_chunks(batch_chunks, boilerplate_stopwords=effective_stopwords)
    if clusters and not dry_run:
        memberships = [
            (cl["cluster_id"], cid, cl["created_at"])
            for cl in clusters
            for cid in cl["chunk_ids"]
        ]
        db.append_clusters(clusters, memberships)

    if merge_samples:
        topics = db.load_cluster_topics(list({s["cluster_id"] for s in merge_samples}))
        for sample in merge_samples:
            sample["cluster_topic"] = topics.get(sample["cluster_id"], "")

    if dry_run:
        # No write happened, so "still stale" cannot be re-queried from the
        # DB -- it is derived from the SAME in-memory decisions that would
        # have driven the writes: every chunk that matched an existing
        # cluster (merged_count) plus every chunk cluster_chunks() actually
        # placed in a NEW self-batch cluster (a candidate list is not
        # itself a guarantee -- MIN_CLUSTER_SIZE can leave chunks out of it).
        newly_self_batch = sum(len(cl["chunk_ids"]) for cl in clusters) if clusters else 0
        chunks_clustered_this_run = merged_count + newly_self_batch
        still_stale = total_stale - chunks_clustered_this_run
    else:
        still_stale_ids = set(stale_transcript_chunk_ids(db))
        # Mark only batch members that FAILED to cluster this run -- a chunk
        # that succeeded is no longer stale and will never be reselected, so
        # it needs no attempt row (see the table's own comment in storage.py).
        failed_this_run = [cid for cid in batch_ids if cid in still_stale_ids]
        if failed_this_run:
            db.mark_recluster_attempted(failed_this_run, run_id=uuid.uuid4().hex[:12])
        still_stale = len(still_stale_ids)
        chunks_clustered_this_run = total_stale - still_stale

    return {
        "refused": False,
        "total_stale_at_start": total_stale,
        "batches_run": 1,
        "chunks_clustered": chunks_clustered_this_run,
        "still_stale": still_stale,
        "fresh_in_batch": fresh_count,
        "fallback_in_batch": fallback_count,
        "merged_into_existing": merged_count,
        "batch_boilerplate_dropped": batch_boilerplate_dropped,
        "merge_samples": merge_samples,
        "merge_run_id": merge_run_id if (merged_count and not dry_run) else None,
        "dry_run": dry_run,
        "drain_command": _drain_command(still_stale),
    }
