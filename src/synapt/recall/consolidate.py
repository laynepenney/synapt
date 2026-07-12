"""Memory consolidation — cross-session knowledge extraction.

Reads enriched journal entries, clusters related sessions, and uses
an LLM to distill durable knowledge patterns. Produces KnowledgeNodes
stored in knowledge.jsonl and indexed in SQLite.

Analogous to sleep consolidation in human memory: episodic memories
(journal entries) are replayed and compressed into semantic memory
(knowledge nodes).

Requires mlx-lm (pip install mlx-lm). Degrades gracefully if not installed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from synapt.recall.journal import (
    JournalEntry,
    _journal_path,
    _read_all_entries,
    _dedup_entries,
)
from synapt.recall.knowledge import (
    KnowledgeNode,
    VALID_CATEGORIES,
    _knowledge_path,
    append_node,
    batch_update_nodes,
    dedup_knowledge_nodes,
    read_nodes,
    compute_confidence,
    update_node,
)
from synapt.recall.clustering import _jaccard
from synapt.recall.scrub import scrub_text, strip_markdown_formatting
from synapt.recall.core import project_data_dir, project_index_dir
from synapt.recall._llm_util import truncate_at_word as _tw
from synapt_extract import create_extraction_builder, finalize_extraction, FinalizeContext

logger = logging.getLogger("synapt.recall.consolidate")


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value not in {"", "0", "false", "no", "off"}

from synapt._models.base import Message
from synapt.recall._mlx import MLX_AVAILABLE as _MLX_AVAILABLE, INSTALL_MSG as _INSTALL_MSG  # noqa: F401

from synapt.recall._model_router import DEFAULT_DECODER_MODEL as DEFAULT_MODEL
MAX_EXISTING_KNOWLEDGE_CHARS = 4000
MAX_JOURNAL_CLUSTER_CHARS = 3000

# Dynamic response budget — no artificial cap.  The model stops at EOS
# naturally; the budget just prevents mid-JSON truncation.
CONTEXT_BUDGET = 8000    # Conservative for 3B quality (32K window, degrades beyond ~8K)
MIN_RESPONSE_TOKENS = 800

# Regex patterns that detect generic programming advice (not project-specific).
# Compiled once at import time for performance.
_GENERIC_PATTERNS = [
    re.compile(p) for p in [
        # "Always use X" / "Never do Y" generic advice
        r"(?i)^(always |never )?(use|write|keep|follow|maintain) (a )?(consistent|clean|good|proper|clear)",
        r"(?i)^(always |never )?(use|write) (unit |integration )?tests\b",
        r"(?i)^(always |never )?(use|prefer) docker\b",
        r"(?i)^(always |never )?(use|follow) (best practices|coding standards?|style guides?)\b",
        r"(?i)^(always |never )?(document|comment) (your |the )?(code|functions)\b",
        r"(?i)^(always |never )?(use|prefer) (version control|git)\s*$",
        r"(?i)^(always |never )?(keep|write) (code|functions|methods) (short|small|simple|clean)\b",
        r"(?i)^(always |never )?use gpu\b(?!.*\b(a100|a10g|l4|t4|h100)\b)",
        r"(?i)^(always |never )?use a? ?consistent naming",
        # Tool-tautology: "Use [tool] for/to [primary purpose]" with NO extra
        # specificity signals.  The negative lookahead prevents false positives
        # when the sentence includes flags, paths, packages, or versions.
        r"(?i)^use (gradlew?|gradle) (for|to) (build|compil|runn?)\w*\b(?!.*( -\w|/|@|\[|:|\d+\.\d))",
        r"(?i)^use (npm|yarn|pnpm|bun) (for|to) (install|manag|runn?)\w*\b(?!.*( -\w|/|@|\[|:|\d+\.\d))",
        r"(?i)^use (pip|poetry|uv|conda) (for|to) (install|manag)\w*\b(?!.*( -\w|/|@|\[|:|\d+\.\d))",
        r"(?i)^use (git|github|gitlab) (for|to) (track|manag|version|stor)\w*\b(?!.*( -\w|/|@|\[|:|\d+\.\d))",
        r"(?i)^use (make|cmake|bazel) (for|to) (build|compil)\w*\b(?!.*( -\w|/|@|\[|:|\d+\.\d))",
        r"(?i)^use (pytest|jest|mocha|junit) (for|to) (test|run tests|testing)\b(?!.*( -\w|/|@|\[|:|\d+\.\d))",
        r"(?i)^use (eslint|flake8|ruff|pylint|clippy) (for|to) (lint|check|format)\w*\b(?!.*( -\w|/|@|\[|:|\d+\.\d))",
        r"(?i)^use (prettier|black|gofmt|rustfmt) (for|to) (format|styl)\w*\b(?!.*( -\w|/|@|\[|:|\d+\.\d))",
        # Generic config/setup knowledge (no specific details)
        r"(?i)^(use|configure|set up) (settings\.gradle|build\.gradle|package\.json|pyproject\.toml|cargo\.toml)\s+(for|to)\b(?!.*( -\w|/|@|\[|:|\d+\.\d))",
        # Generic workflow advice
        r"(?i)^(review|test|verify|validate) (code|changes|pull requests?) before (merging|deploying|releasing)\b",
        r"(?i)^(handle|catch|log) (errors?|exceptions?) (properly|gracefully|carefully)\b",
        r"(?i)^(keep|maintain|update) depend(encies|ency) (up.to.date|current|regularly)\b",
    ]
]

# Code-specific generic patterns — tool output noise that passes the general
# filter because it's concrete, but adds no retrieval value.  Only applied
# when content_type is "code".
_CODE_GENERIC_PATTERNS = [
    re.compile(p) for p in [
        r"(?i)^(build|compilation) (succeeded|completed|finished|passed)\b",
        r"(?i)^all (tests?|checks?) (pass(ed)?|succeeded|green)\b",
        r"(?i)^(file|changes?) saved\b",
        r"(?i)^(linting|formatting|type.?check) (passed|clean|ok)\b",
        r"(?i)^no (errors?|warnings?|issues?) found\b",
        r"(?i)^(installed|updated|removed) \d+ packages?\b",
        r"(?i)^(successfully )?(deployed|published|released|uploaded)\b(?!.*\bv?\d+\.\d)",
    ]
]

# Specificity signals — content with these patterns is likely project-specific.
# Presence of ANY of these exempts a node from the low-specificity filter.
# Note: no IGNORECASE — CamelCase detection requires case sensitivity.
_SPECIFICITY_SIGNALS = re.compile(
    r"/[\w./-]{3,}"              # File paths
    r"|v\d+\.\d+"                # Version numbers
    r"|\d+\.\d+\.\d+"           # SemVer
    r"|--[\w-]{2,}"              # CLI flags
    r"|\b[A-Z][a-z]+[A-Z]\w*"   # CamelCase identifiers
    r"|\b[A-Z]\d{2,}\w*\b"      # Model/hardware identifiers: A100, H100, L4, T4
    r"|\b[a-z]\w+_\w+\b"        # Snake_case identifiers (2+ parts, lowercase start)
    r"|\b[Ss]ession\s*#?\d+"    # Session references
    r"|\b[Pp][Rr]\s*#?\d+"      # PR references
    r"|\b[Ii]ssue\s*#?\d+"      # Issue references
    r"|\b[Cc]onv\s*#?\d+"       # Conv references
    r"|\b\d{4}-\d{2}-\d{2}\b"   # Dates
    r"|\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d"  # Month+day/year
    ,
)

# Proper nouns not at start of sentence indicate entity-specific content.
# Matches capitalized words (3+ chars) that aren't the first word and aren't
# common English words. Applied separately since it needs word-position context.
_COMMON_CAPS = frozenset({
    # Pronouns & determiners
    "The", "This", "That", "These", "Those", "They", "Their", "There",
    "What", "When", "Where", "Which", "While", "Who", "Why", "How",
    "And", "But", "For", "Not", "All", "Any", "Some", "From", "With",
    "Into", "Also", "Just", "Very", "Each", "Both", "After", "Before",
    "During", "About", "Above", "Below", "Between", "Through",
    # Common verbs at sentence start
    "Use", "Using", "Used", "Keep", "Always", "Never", "Follow",
    "Make", "Run", "Set", "Get", "Let", "See", "Try", "Add",
    "Write", "Read", "Check", "Test", "Build", "Create", "Store",
    "Handle", "Configure", "Install", "Deploy", "Start", "Stop",
    "Enable", "Disable", "Include", "Avoid", "Ensure", "Verify",
    "Review", "Update", "Remove", "Move", "Copy", "Save", "Load",
    "Define", "Implement", "Consider", "Prefer", "Maintain",
    "Monitor", "Optimize", "Migrate", "Document", "Refactor",
    "Debug", "Validate", "Integrate", "Manage", "Process",
    # Common technology names (not project-specific)
    "Docker", "Gradle", "Android", "Python", "Java", "Swift",
    "Rust", "Linux", "Windows", "React", "Node", "Rails",
    "Redis", "Mongo", "Postgres", "MySQL", "Nginx", "Apache",
    "Kubernetes", "Terraform", "Jenkins", "Github", "Gitlab",
})


def _has_proper_nouns(content: str) -> bool:
    """Check if content contains proper nouns (named entities).

    Finds capitalized words that aren't common English words. The first
    word is checked too but only counted if it's not a common sentence
    starter. One proper noun is sufficient — it indicates the content
    refers to a specific entity (person, place, product).
    """
    words = content.split()
    if len(words) < 2:
        return False
    for w in words:
        clean = w.rstrip(".,;:!?'\")")
        if (len(clean) >= 2
                and clean[0].isupper()
                and clean[1:].islower()
                and clean not in _COMMON_CAPS):
            return True
    return False

def _lacks_specificity(
    content: str,
    threshold: int = 120,
    content_type: str | None = None,
) -> bool:
    """Return True if content lacks project-specific identifiers.

    Catches tool-knowledge that's technically accurate but not specific
    to the project — e.g., "Use gradlew and settings.gradle.kts for
    root Gradle builds" is true for ALL Gradle projects.

    Only applied to short content (< threshold chars) where specificity
    signals are more meaningful. Longer content is more likely to include
    context even without explicit identifiers.

    Args:
        content: Text to check.
        threshold: Character length above which the filter is skipped.
            Default 120; set higher to catch more, or 10000+ to disable.
        content_type: "code", "personal", or "mixed". When "code",
            additional code-specific generic patterns are checked.
    """
    if len(content) > threshold:
        return False
    if _SPECIFICITY_SIGNALS.search(content) is not None:
        return False
    # Proper nouns (person names, place names) are strong entity signals
    if _has_proper_nouns(content):
        return False
    # Code-specific noise: short tool output that looks concrete but
    # adds no retrieval value (e.g., "Build succeeded", "All tests pass")
    if content_type == "code":
        for pat in _CODE_GENERIC_PATTERNS:
            if pat.search(content):
                return True
    return True

# Stopwords for keyword extraction
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "and", "but", "or", "if", "while", "that", "this", "these", "those",
    "it", "its", "i", "we", "you", "he", "she", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "our", "their", "what",
    "which", "who", "whom", "up", "out", "about", "just", "also",
    "new", "used", "using", "use", "added", "add", "fixed", "fix",
    "file", "files", "code", "session", "work", "working",
})


CONSOLIDATION_PROMPT = """\
You are analyzing session summaries to extract durable, specific knowledge.

## Project Context
{project_context}

## Existing Knowledge
{existing_knowledge}

## Recent Sessions
{journal_cluster}

## Examples of GOOD knowledge nodes (specific, concrete):
{good_examples}
(These show the FORMAT only. Do NOT copy these examples. Extract facts from the sessions above, using the real names and details found there.)

## Examples of BAD knowledge nodes (generic — NEVER produce these):
- "Always use Docker for containerization"
- "Use a consistent naming convention"
- "Write tests before deploying"
- "Use GPU for training"
- "Document your code thoroughly"

## Task
Extract patterns that represent durable knowledge — things true across sessions, not one-off observations.

Categories (use the best fit):
- fact: specific names, dates, relationships, numbers, personal details
- preference: stated likes, dislikes, choices about food, hobbies, etc.
- decision: explicit choices made between alternatives
- convention: agreed-upon patterns or rules
- workflow: recurring processes or routines
- architecture: system structure or design choices
- infrastructure: hosting, hardware, config values
- tooling: specific tools, versions, or setup
- debugging: diagnosed root causes or fixes
- lesson-learned: insights from mistakes or surprises

Rules:
1. Extract patterns that appear across sessions OR strongly-stated specific facts (names, preferences, relationships, config values).
2. If a newer session REVERSES a decision from an older session, mark it as a contradiction. Produce the NEW fact with "action": "contradict" and reference the old node's ID.
3. If a pattern matches an existing knowledge node, use "action": "corroborate" with the existing node's ID.
4. Keep each fact concise (1-2 sentences, max 200 chars).
5. Be concrete and specific — include specific names, values, paths, or details from the sessions.
6. Do NOT extract generic advice that could apply to any project. Every node must be grounded in the sessions above.
7. Prefer extracting granular personal details that would be hard to find later: nicknames, specific possessions, hobbies, places visited, family members, physical descriptions, stated opinions, specific dates/events.
8. Confidence guide: 0.7-0.9 = verified across 2+ sessions or very explicitly stated; 0.4-0.6 = from a single session with reasonable certainty; below 0.4 = inferred or speculative.
9. If no specific patterns emerge, output {{"nodes": []}}. Empty is better than generic.

Output ONLY valid JSON, no markdown fences, no explanation:
{{"nodes": [{{"action": "create", "existing_id": null, "content": "...", "category": "...", "confidence": 0.6, "tags": ["tag1"], "contradiction_note": "", "source_turns": ["s001c00:5", "s003c00:12"], "valid_from": null, "valid_until": null}}]}}

source_turns: list the session:turn pairs where this fact appears. Format: "session_id:turn_number". Include ALL turns that support the fact — these are used to link back to the original conversation.

valid_from/valid_until: ISO 8601 dates (e.g. "2026-03-15") or null. Set these when the fact has clear temporal boundaries:
- "We migrated to PostgreSQL in March 2026" → valid_from: "2026-03-01"
- "The API key expires April 30" → valid_until: "2026-04-30"
- "Before the refactor, we used callbacks" → valid_until: (date of refactor session)
- Timeless facts (names, preferences) → both null
Only set dates you're confident about. null is better than guessing.

If no durable patterns emerge, output: {{"nodes": []}}
"""

COLLECTION_EXTRACTION_PROMPT = """\
You are analyzing knowledge nodes to find entity collections — groups of similar things \
mentioned across multiple sessions.

## Existing Knowledge Nodes
{existing_knowledge}

## Task
Find entity collections: groups of 3+ similar items scattered across different sessions. \
For each collection, produce a single summary node that enumerates all instances.

Examples of collections:
- "User has 5 model kits: Revell F-15 (s3), Tamiya Spitfire (s7), B-29 bomber (s12), '69 Camaro (s14), Gundam RX-78 (s19)"
- "User visited 4 countries in 2025: Japan (s5), Italy (s8, s9), Brazil (s15), Iceland (s22)"
- "Project uses 3 databases: PostgreSQL for users (s2), Redis for cache (s4), Elasticsearch for search (s11)"

Rules:
1. Only create collections from entities ALREADY in existing knowledge nodes — do not invent items.
2. Each collection must reference 3+ items from 2+ different sessions.
3. Include the specific names/details and source sessions for each item.
4. Category is always "collection".
5. If an existing collection node covers the same entities, use "action": "corroborate" with its ID. \
   If the existing collection is missing items, use "action": "contradict" to supersede it with a complete list.
6. If no collections are found, output {{"nodes": []}}.

Output ONLY valid JSON:
{{"nodes": [{{"action": "create", "existing_id": null, "content": "...", "category": "collection", \
"confidence": 0.8, "tags": ["collection-type"], "source_turns": [], "valid_from": null, "valid_until": null}}]}}
"""

CONSOLIDATION_PROMPT_MINIMAL = """\
## Project Context
{project_context}

## Existing Knowledge
{existing_knowledge}

## Recent Sessions
{journal_cluster}

Categories: fact (names/dates/details), preference, decision, convention, workflow, architecture, infrastructure, tooling, debugging, lesson-learned

Extract durable knowledge as JSON. Be specific — include names, values, dates. Include source_turns citing session:turn pairs. Include valid_from/valid_until (ISO dates or null) when facts have clear temporal boundaries. Output ONLY valid JSON:
{{"nodes": [{{"action": "create|corroborate|contradict", "existing_id": null, "content": "...", "category": "...", "confidence": 0.6, "tags": ["tag1"], "contradiction_note": "", "source_turns": ["s001c00:5"], "valid_from": null, "valid_until": null}}]}}
"""


@dataclass
class ConsolidationResult:
    """Summary of a consolidation run."""
    nodes_created: int = 0
    nodes_corroborated: int = 0
    nodes_contradicted: int = 0
    nodes_deduped: int = 0
    entries_processed: int = 0
    clusters_found: int = 0


def _validate_iso_date(value: str | None) -> str | None:
    """Validate an ISO 8601 date string from LLM output.

    Returns the value if it parses as a valid date, None otherwise.
    Handles full ISO timestamps and date-only strings.

    >>> _validate_iso_date("2026-03-15")
    '2026-03-15'
    >>> _validate_iso_date("2026-03-15T10:00:00Z")
    '2026-03-15T10:00:00Z'
    >>> _validate_iso_date("March 2026")
    >>> _validate_iso_date(None)
    """
    if not value or not isinstance(value, str):
        return None
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value
    except (ValueError, AttributeError):
        return None


def _cluster_valid_from(cluster: list[JournalEntry]) -> str | None:
    """Return the earliest parseable journal timestamp in *cluster*."""
    earliest: tuple[datetime, str] | None = None
    for entry in cluster:
        if not entry.timestamp:
            continue
        try:
            dt = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        if earliest is None or dt < earliest[0]:
            earliest = (dt, entry.timestamp)
    return earliest[1] if earliest is not None else None


def _is_generic_node(content: str) -> bool:
    """Return True if content matches a known generic advice pattern."""
    for pattern in _GENERIC_PATTERNS:
        if pattern.search(content):
            return True
    return False


# Patterns that detect garbled content where raw LLM structural metadata
# leaked into the content field. Common with 3B models that fail to follow
# the JSON schema and mix formatting/metadata into their output.
_GARBLED_CONTENT_PATTERNS = [
    re.compile(p) for p in [
        # Source turn references stored as content
        r"^Source turns?:\s",
        # Raw LLM output with inline metadata (existing_id, contradiction_note)
        r"\bexisting_id:\s",
        r"\bcontradiction_note:\s",
        # Content that's just a quoted duplicate with metadata annotation
        r'^"[^"]+"\s+\((fact|preference|decision|convention|workflow|lesson-learned),\s+(corroborat|contradict)',
        # Content that wraps another content string in quotes with category
        r'^"[^"]+"\s+\([^)]*confidence:\s',
        # Verified/corroborate annotations stored as content
        r"\(fact\)\s*-\s*verified",
    ]
]


def _is_garbled_content(content: str) -> bool:
    """Return True if content looks like garbled LLM structural metadata."""
    for pattern in _GARBLED_CONTENT_PATTERNS:
        if pattern.search(content):
            return True
    return False


# Pattern for section header prefixes that 3B models inject before content:
# "LGBTQ+ Support: Caroline received...", "Art as Self-Expression: ..."
# Only strip when the prefix is short (2-5 words) and the rest is a sentence.
_SECTION_PREFIX_RE = re.compile(
    r"^([A-Z][\w\s&+/-]{2,40}):\s+([A-Z])"
)


def _strip_section_prefix(content: str) -> str:
    """Strip topic-label prefixes that 3B models inject before content.

    Examples:
        "LGBTQ+ Support: Caroline received support" → "Caroline received support"
        "Art as Self-Expression: Caroline discussed art" → "Caroline discussed art"
        "API: Use POST /api/users" → kept as-is (technical content)
    """
    m = _SECTION_PREFIX_RE.match(content)
    if m:
        prefix = m.group(1).strip()
        # Only strip if prefix looks like a topic label (no technical terms)
        # and the remainder is a real sentence (>20 chars)
        remainder = content[m.end() - 1:]  # Include the capital letter
        prefix_words = prefix.split()
        if len(remainder) > 20 and 2 <= len(prefix_words) <= 5:
            return remainder
    return content


# Default GOOD examples used when no existing knowledge nodes are available.
# These are replaced dynamically by the project's own nodes when they exist.
# IMPORTANT: These must be clearly generic/hypothetical so small models don't
# parrot them as actual knowledge. Each is prefixed with "Example:" and uses
# deliberately vague hypothetical phrasing.
_DEFAULT_GOOD_EXAMPLES = [
    '- FORMAT EXAMPLE: "[PersonA] prefers herbal tea over coffee" (preference)',
    '- FORMAT EXAMPLE: "[PersonB] adopted a rescue dog named Rex in April 2025" (fact)',
    '- FORMAT EXAMPLE: "[PersonA] calls [PersonB] by the nickname Zee" (fact)',
    '- FORMAT EXAMPLE: "[PersonB] grew up in Dublin, Ireland before moving abroad" (fact)',
    '- FORMAT EXAMPLE: "[PersonA] switched from yoga to pilates for back pain" (decision)',
    '- FORMAT EXAMPLE: "[PersonB] collects classic children\'s books, especially Dr. Seuss" (preference)',
    '- FORMAT EXAMPLE: "[PersonA] is studying for a counseling certification" (fact)',
    '- FORMAT EXAMPLE: "[PersonB] enjoys classical music, particularly Vivaldi" (preference)',
]


def _build_few_shot_examples(
    existing_nodes: list[KnowledgeNode],
    max_examples: int = 4,
) -> str:
    """Build GOOD few-shot examples for the consolidation prompt.

    When existing knowledge nodes are available, uses the top nodes
    (one per category for diversity) as examples. Falls back to
    hardcoded defaults for new projects with no existing knowledge.
    """
    if not existing_nodes:
        return "\n".join(_DEFAULT_GOOD_EXAMPLES)
    # Pick the highest-confidence node per category for diversity
    by_category: dict[str, KnowledgeNode] = {}
    for node in existing_nodes:
        if node.category not in by_category or node.confidence > by_category[node.category].confidence:
            by_category[node.category] = node
    selected = sorted(
        by_category.values(), key=lambda n: n.confidence, reverse=True,
    )[:max_examples]
    return "\n".join(f'- "{n.content}" ({n.category})' for n in selected)


def _get_project_context(project_dir: Path) -> str:
    """Build project context string for the consolidation prompt."""
    name = project_dir.name
    description = ""
    claude_md = project_dir / "CLAUDE.md"
    if claude_md.exists():
        try:
            for line in claude_md.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                # Skip headings and blank lines, find first content line
                if line and not line.startswith("#") and not line.startswith("```"):
                    description = line[:200]
                    break
        except OSError:
            pass
    parts = [f"Project: {name}"]
    if description:
        parts.append(f"Description: {description}")
    return "\n".join(parts)


def _extract_keywords(text: str) -> set[str]:
    """Extract non-trivial keywords from text for clustering."""
    words = re.findall(r"[a-z][a-z0-9_.-]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


# Cached embedding provider for inline dedup (loaded once per process).
_inline_emb_provider = None
_inline_emb_loaded = False

_INLINE_COSINE_THRESHOLD = 0.80

# Content-type-aware dedup thresholds.
# Personal content gets more permissive thresholds (preserve nuanced facts).
# Code content gets more aggressive thresholds (filter generic tool output).
_DEDUP_THRESHOLDS: dict[str, tuple[float, float]] = {
    # (jaccard_threshold, cosine_threshold)
    "personal": (0.6, 0.88),
    "code": (0.4, 0.75),
    "mixed": (0.5, 0.80),
}


def _get_dedup_thresholds(content_profile=None) -> tuple[float, float]:
    """Return (jaccard_threshold, cosine_threshold) for the given content type."""
    if content_profile is None:
        return _DEDUP_THRESHOLDS["mixed"]
    ct = getattr(content_profile, "content_type", "mixed")
    return _DEDUP_THRESHOLDS.get(ct, _DEDUP_THRESHOLDS["mixed"])


def _inline_embedding_dedup(
    candidate_content: str,
    existing_nodes: "list[KnowledgeNode]",
    threshold: float = _INLINE_COSINE_THRESHOLD,
) -> "tuple[KnowledgeNode | None, float]":
    """Check if candidate is a semantic duplicate of any existing node.

    Uses the embedding provider (cached per process) to compute cosine
    similarity.  Returns (best_match_node, cosine_similarity) if above
    threshold, else (None, 0.0).  Degrades gracefully if embeddings are
    unavailable.
    """
    global _inline_emb_provider, _inline_emb_loaded
    if not _inline_emb_loaded:
        _inline_emb_loaded = True
        try:
            from synapt.recall.embeddings import get_embedding_provider
            _inline_emb_provider = get_embedding_provider()
        except Exception:
            pass

    if _inline_emb_provider is None or not existing_nodes:
        return (None, 0.0)

    try:
        from synapt.recall.embeddings import cosine_similarity

        # Batch embed: candidate + all existing
        texts = [candidate_content] + [n.content for n in existing_nodes]
        embeddings = _inline_emb_provider.embed(texts)
        cand_emb = embeddings[0]

        best_match = None
        best_sim = 0.0
        for i, node in enumerate(existing_nodes):
            sim = cosine_similarity(cand_emb, embeddings[i + 1])
            if sim > best_sim:
                best_sim = sim
                best_match = node

        if best_sim >= threshold:
            return (best_match, best_sim)
    except Exception:
        logger.debug("Inline embedding dedup failed", exc_info=True)

    return (None, 0.0)


def _dedup_decisions_path(project_dir: Path | None = None) -> Path:
    """Return path to dedup_decisions.jsonl in the project's .synapt/recall/ dir."""
    return project_data_dir(project_dir) / "dedup_decisions.jsonl"


# ---------------------------------------------------------------------------
# Cluster-level LLM response cache
# ---------------------------------------------------------------------------

def _cluster_cache_key(cluster: list[JournalEntry]) -> str:
    """Deterministic cache key from a cluster's entries."""
    parts = sorted(f"{e.session_id}|{e.timestamp}" for e in cluster)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _load_response_cache(cache_path: Path) -> dict[str, dict]:
    """Load cached LLM responses keyed by cluster hash.

    Returns dict mapping cache key → {"response": str, "prompt": str}.
    """
    cache: dict[str, dict] = {}
    if not cache_path.exists():
        return cache
    try:
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    cache[d["key"]] = {
                        "response": d["response"],
                        "prompt": d.get("prompt", ""),
                    }
                except (json.JSONDecodeError, KeyError):
                    continue
    except OSError:
        pass
    return cache


def _save_cached_response(
    cache_path: Path, key: str, response: str, prompt: str = "",
) -> None:
    """Append a successful LLM response (and prompt) to the cache.

    Stores both prompt and response so each entry is a complete
    training pair for a future consolidation adapter.
    """
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"key": key, "response": response}
        if prompt:
            entry["prompt"] = prompt
        from synapt.recall._filelock import lock_exclusive, unlock
        with open(cache_path, "a", encoding="utf-8") as f:
            lock_exclusive(f)
            try:
                f.write(json.dumps(entry) + "\n")
                f.flush()
            finally:
                unlock(f)
    except OSError:
        pass  # Cache is best-effort


def _log_dedup_decision(
    decision_path: Path,
    *,
    action: str,
    candidate_content: str,
    candidate_category: str,
    session_ids: list[str] | None = None,
    existing_id: str = "",
    existing_content: str = "",
    similarity_score: float | None = None,
    source: str = "",
    contradiction_note: str = "",
    negative_pairs: list[dict] | None = None,
) -> None:
    """Append one pairwise decision to the dedup decisions JSONL file.

    Pure logging — never disrupts consolidation.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "candidate_content": candidate_content,
        "candidate_category": candidate_category,
        "session_ids": session_ids or [],
        "source": source,
    }
    if existing_id:
        entry["existing_id"] = existing_id
    if existing_content:
        entry["existing_content"] = existing_content
    if similarity_score is not None:
        entry["similarity_score"] = round(similarity_score, 4)
    if contradiction_note:
        entry["contradiction_note"] = contradiction_note
    if negative_pairs:
        entry["negative_pairs"] = negative_pairs

    try:
        from synapt.recall._filelock import lock_exclusive
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        with open(decision_path, "a", encoding="utf-8") as f:
            lock_exclusive(f)
            f.write(json.dumps(entry) + "\n")
            f.flush()
    except OSError:
        logger.debug("Failed to write dedup decision log")


def _entry_keywords(entry: JournalEntry) -> set[str]:
    """Extract keywords from all text fields of a journal entry."""
    parts = [entry.focus or ""]
    parts.extend(entry.done or [])
    parts.extend(entry.decisions or [])
    parts.extend(entry.next_steps or [])
    return _extract_keywords(" ".join(parts))


def cluster_journal_entries(
    entries: list[JournalEntry],
) -> list[list[JournalEntry]]:
    """Group related journal entries by file overlap and keyword overlap.

    Uses union-find: two entries are related if Jaccard(files) > 0.3
    OR they share 2+ non-trivial keywords. Connected components become
    clusters. Singletons (no overlap) are discarded.

    When file/keyword clustering produces no clusters (common for
    conversational data without code changes), falls back to temporal
    windowing — groups entries into consecutive pairs by timestamp.
    """
    n = len(entries)
    if n < 2:
        return []

    # Build file sets and keyword sets
    file_sets = [set(e.files_modified or []) for e in entries]
    keyword_sets = [_entry_keywords(e) for e in entries]

    # Union-find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            # File overlap
            if file_sets[i] and file_sets[j] and _jaccard(file_sets[i], file_sets[j]) > 0.3:
                union(i, j)
                continue
            # Keyword overlap (2+ shared)
            shared = keyword_sets[i] & keyword_sets[j]
            if len(shared) >= 2:
                union(i, j)

    # Group by root
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Return only clusters with 2+ entries, splitting large ones
    clusters = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        group = [entries[i] for i in indices]
        clusters.extend(_split_large_cluster(group))

    # Fallback: when no file/keyword clusters found, use temporal windows.
    # This handles conversational data (no code, no files) where entries
    # still contain useful knowledge that should be consolidated.
    if not clusters and n >= 2:
        logger.info(
            "No file/keyword clusters found among %d entries; "
            "falling back to temporal windowing",
            n,
        )
        clusters = _temporal_window_clusters(entries)

    return clusters


def _temporal_window_clusters(
    entries: list[JournalEntry],
    window_size: int = 3,
) -> list[list[JournalEntry]]:
    """Group entries into overlapping temporal windows.

    Reuses ``_split_large_cluster`` which implements the same sliding-
    window algorithm with 1-entry overlap.
    """
    if len(entries) < 2:
        return []
    return _split_large_cluster(entries, max_size=window_size)


def _split_large_cluster(
    cluster: list[JournalEntry],
    max_size: int = 4,
) -> list[list[JournalEntry]]:
    """Split a large cluster into time-ordered sub-clusters.

    Union-find can produce mega-clusters via transitive chaining (A-B-C-...).
    A 27-entry cluster formatted to 3000 chars only shows ~5 entries to the
    model. Splitting into windows of max_size ensures the model sees all
    entries across multiple LLM calls.

    Sub-clusters overlap by 1 entry to preserve cross-window context.
    """
    # Sort by timestamp so windows are temporally coherent
    ordered = sorted(cluster, key=lambda e: e.timestamp or "")
    if len(ordered) <= max_size:
        return [ordered]
    step = max_size - 1  # overlap of 1 entry between windows
    sub_clusters = []
    for start in range(0, len(ordered), step):
        window = ordered[start : start + max_size]
        if len(window) >= 2:
            sub_clusters.append(window)
    return sub_clusters


def _format_existing_knowledge(
    nodes: list[KnowledgeNode],
    cluster: list[JournalEntry] | None = None,
    max_relevant: int = 8,
) -> str:
    """Format existing nodes for the consolidation prompt.

    When a cluster is provided, ranks nodes by keyword relevance to the
    cluster and includes only the top ``max_relevant`` nodes.  This keeps
    the prompt focused and leaves more token budget for the response.

    Nodes with keyword overlap appear first; remaining slots are filled
    with the highest-confidence nodes (they may be needed for cross-topic
    corroboration).
    """
    if not nodes:
        return "(none yet)"

    # Without a cluster, fall back to the original truncation behaviour
    if cluster is None:
        lines = [f"[{n.id}] ({n.category}) {n.content}" for n in nodes]
        text = "\n".join(lines)
        if len(text) > MAX_EXISTING_KNOWLEDGE_CHARS:
            text = text[:MAX_EXISTING_KNOWLEDGE_CHARS] + "\n... (truncated)"
        return text

    # Score each node by keyword overlap with the cluster.
    # Uses overlap coefficient |A∩B|/|B| (B = node keywords) instead of
    # Jaccard — cluster keyword sets are much larger than node sets, so
    # Jaccard would dilute all scores toward zero.
    cluster_kw: set[str] = set()
    for entry in cluster:
        cluster_kw |= _entry_keywords(entry)

    scored = []
    for node in nodes:
        node_kw = _extract_keywords(node.content)
        if node_kw:
            sim = len(cluster_kw & node_kw) / len(node_kw)
        else:
            sim = 0.0
        scored.append((sim, node.confidence, node))

    # Sort: highest relevance first, then highest confidence as tiebreaker
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

    selected = scored[:max_relevant]
    omitted = len(scored) - len(selected)

    lines = [f"[{n.id}] ({n.category}) {n.content}" for _, _, n in selected]
    if omitted > 0:
        lines.append(
            f"... ({omitted} more active nodes not relevant to this cluster)"
        )
    return "\n".join(lines)


def _detect_concurrent_agents(cluster: list[JournalEntry]) -> str:
    """Detect if a cluster contains concurrent sessions from different agents.

    Returns annotation text for the LLM prompt if concurrent agents are
    detected, empty string otherwise.  Concurrent = timestamps within ~1 hour
    of each other AND different griptree identities.
    """
    agents = {getattr(e, "griptree", "") for e in cluster if getattr(e, "griptree", "")}
    if len(agents) < 2:
        return ""

    # Check timestamp overlap: sort by time, see if any pair is within ~1 hour
    timed = sorted(
        ((e.timestamp or "", getattr(e, "griptree", "") or e.session_id[:8]) for e in cluster if e.timestamp),
        key=lambda t: t[0],
    )
    concurrent_pairs: list[tuple[str, str]] = []
    for i in range(len(timed)):
        for j in range(i + 1, len(timed)):
            if timed[i][1] == timed[j][1]:
                continue  # Same agent
            # Quick ISO timestamp comparison: within ~30 min
            # ISO 8601 sorts lexicographically, so we check the date+hour
            t1, t2 = timed[i][0][:16], timed[j][0][:16]  # "YYYY-MM-DDTHH:MM"
            if t1[:13] == t2[:13]:  # Same hour
                concurrent_pairs.append((timed[i][1], timed[j][1]))
            elif t1[:10] == t2[:10]:  # Same day, check if adjacent hours
                try:
                    h1, h2 = int(t1[11:13]), int(t2[11:13])
                    if abs(h1 - h2) <= 1:
                        concurrent_pairs.append((timed[i][1], timed[j][1]))
                except (ValueError, IndexError):
                    pass

    if not concurrent_pairs:
        return ""

    agent_list = ", ".join(sorted(agents))
    return (
        f"NOTE: This cluster contains CONCURRENT sessions from {len(agents)} "
        f"agents ({agent_list}) working in parallel on related tasks. "
        f"These are collaborative, not sequential — facts may be corroborated "
        f"across agents or represent parallel discoveries."
    )


def _format_journal_cluster(cluster: list[JournalEntry]) -> str:
    """Format a cluster of journal entries for the consolidation prompt.

    Includes agent identity when available and annotates concurrent
    multi-agent sessions so the LLM understands collaborative context.
    """
    lines = []

    # Add concurrency annotation if multiple agents detected
    concurrency_note = _detect_concurrent_agents(cluster)
    if concurrency_note:
        lines.append(concurrency_note)
        lines.append("")

    for entry in sorted(cluster, key=lambda e: e.timestamp):
        sid = entry.session_id[:8] if entry.session_id else "unknown"
        date = entry.timestamp[:10] if entry.timestamp else "?"
        agent = getattr(entry, "griptree", "") or ""
        agent_label = f" ({agent})" if agent else ""
        parts = [f"[Session {sid}{agent_label}, {date}]"]
        if entry.focus:
            parts.append(f"Focus: {entry.focus}")
        if entry.done:
            parts.append(f"Done: {'; '.join(entry.done)}")
        if entry.decisions:
            parts.append(f"Decisions: {'; '.join(entry.decisions)}")
        if entry.next_steps:
            parts.append(f"Next: {'; '.join(entry.next_steps)}")
        lines.append(" | ".join(parts))
    text = "\n".join(lines)
    if len(text) > MAX_JOURNAL_CLUSTER_CHARS:
        text = text[:MAX_JOURNAL_CLUSTER_CHARS] + "\n... (truncated)"
    return text


def _build_consolidation_prompt(
    cluster: list[JournalEntry],
    existing_nodes: list[KnowledgeNode],
    project_dir: Path | None = None,
    adapter_path: str = "",
) -> str:
    """Build the consolidation prompt for an LLM call.

    Uses minimal prompt when *adapter_path* is provided (the adapter has
    learned format/rules from training data).  Uses the full prompt with
    rules and examples for base-model inference.
    """
    ctx = _get_project_context(project_dir) if project_dir else "Project: unknown"
    existing = _format_existing_knowledge(existing_nodes, cluster=cluster)
    journal = _format_journal_cluster(cluster)

    if adapter_path:
        return CONSOLIDATION_PROMPT_MINIMAL.format(
            project_context=ctx,
            existing_knowledge=existing,
            journal_cluster=journal,
        )
    return CONSOLIDATION_PROMPT.format(
        project_context=ctx,
        existing_knowledge=existing,
        journal_cluster=journal,
        good_examples=_build_few_shot_examples(existing_nodes),
    )


def _estimate_response_budget(prompt: str) -> int:
    """Estimate an appropriate ``max_tokens`` for a consolidation LLM call.

    Uses a ``len(prompt) // 4`` heuristic (~4 chars per token — the same
    approximation used in ``core.py``).  No upper cap: the model stops at
    EOS naturally, so a generous budget only matters when the model
    *needs* more output tokens.
    """
    prompt_tokens = len(prompt) // 4
    return max(MIN_RESPONSE_TOKENS, CONTEXT_BUDGET - prompt_tokens)


def _parse_llm_response(response: str) -> dict | None:
    """Parse the LLM's JSON response."""
    from synapt.recall._llm_util import parse_llm_json
    return parse_llm_json(response)


def _apply_consolidation_result(
    parsed: dict,
    existing_nodes: list[KnowledgeNode],
    cluster: list[JournalEntry],
    knowledge_path: Path,
    decision_log_path: Path | None = None,
    db=None,
    content_profile=None,
) -> ConsolidationResult:
    """Apply parsed LLM output: create, corroborate, or contradict nodes.

    When *db* (RecallDB) is provided, contradictions are queued as
    pending_contradictions for user review instead of auto-applied.
    """
    result = ConsolidationResult()
    nodes_list = parsed.get("nodes", [])
    if not isinstance(nodes_list, list):
        return result

    # Collect session_ids from this cluster
    cluster_sessions = [
        e.session_id for e in cluster if e.session_id
    ]

    # Index existing nodes by ID for lookups
    existing_by_id = {n.id: n for n in existing_nodes}

    for raw_node in nodes_list:
        if not isinstance(raw_node, dict):
            continue

        action = raw_node.get("action", "create")
        content = scrub_text(_tw(str(raw_node.get("content", "")), 300))
        # Strip markdown formatting (bold/italic) that small models inject
        content = strip_markdown_formatting(content)
        # Strip section header prefixes ("LGBTQ+ Support: ..." → "...")
        content = _strip_section_prefix(content)
        category = scrub_text(str(raw_node.get("category", "workflow")))
        tags = raw_node.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tags = [scrub_text(str(t)) for t in tags if t]

        # Parse source_turns from LLM output (e.g., ["s001c00:5", "s003c00:12"])
        source_turns = raw_node.get("source_turns", [])
        if not isinstance(source_turns, list):
            source_turns = []
        source_turns = [str(t) for t in source_turns if t]

        if not content:
            continue

        # Get adaptive params for content-aware filtering
        _ap = None
        if content_profile is not None:
            from synapt.recall.content_profile import adaptive_params
            _ap = adaptive_params(content_profile)

        # Reject generic programming advice (disabled for personal content)
        if action == "create" and (_ap is None or _ap.generic_filter_enabled):
            if _is_generic_node(content):
                logger.info("Rejected generic node (pattern): %s", content[:80])
                continue
        # Reject low-specificity (threshold + patterns adapt to content type)
        if action == "create":
            spec_threshold = _ap.specificity_threshold if _ap else 120
            _ct = content_profile.content_type if content_profile is not None else None
            if _lacks_specificity(content, threshold=spec_threshold, content_type=_ct):
                logger.info("Rejected generic node (low specificity): %s", content[:80])
                continue

        # Reject contamination from few-shot example placeholders
        if "[PersonA]" in content or "[PersonB]" in content:
            logger.info("Rejected example-contaminated node: %s", content[:80])
            continue

        # Reject garbled content from 3B parsing failures — raw LLM
        # structural metadata leaked into content.
        if action == "create" and _is_garbled_content(content):
            logger.info("Rejected garbled node: %s", content[:80])
            continue

        if action == "corroborate":
            existing_id = raw_node.get("existing_id", "")
            # 3B models sometimes return a list instead of a string
            if isinstance(existing_id, list):
                existing_id = existing_id[0] if existing_id else ""
            target = existing_by_id.get(existing_id)
            if target:
                # Add new source sessions, bump confidence
                new_sources = list(set(target.source_sessions + cluster_sessions))
                new_confidence = compute_confidence(len(new_sources))
                update_node(
                    target.id,
                    {
                        "source_sessions": new_sources,
                        "confidence": new_confidence,
                    },
                    knowledge_path,
                )
                result.nodes_corroborated += 1
                if decision_log_path:
                    _log_dedup_decision(
                        decision_log_path,
                        action="corroborate",
                        candidate_content=content,
                        candidate_category=category,
                        existing_id=existing_id,
                        existing_content=target.content,
                        source="llm",
                        session_ids=cluster_sessions,
                    )
                continue  # Done with this node
            else:
                # Existing node not found — fall through to create
                action = "create"

        if action == "contradict":
            existing_id = raw_node.get("existing_id", "")
            if isinstance(existing_id, list):
                existing_id = existing_id[0] if existing_id else ""
            contradiction_note = scrub_text(
                _tw(str(raw_node.get("contradiction_note", "")), 200)
            )
            # Reject generic replacement content (pattern-only; no specificity
            # check since contradictions reference existing project-specific nodes)
            if _is_generic_node(content):
                logger.info("Rejected generic contradict node: %s", content[:80])
                continue
            target = existing_by_id.get(existing_id)
            if target and db is not None:
                # Queue for user review instead of auto-applying
                db.add_pending_contradiction(
                    old_node_id=target.id,
                    new_content=content,
                    category=category,
                    reason=contradiction_note,
                    source_sessions=cluster_sessions,
                    detected_by="consolidation",
                )
                result.nodes_contradicted += 1
                logger.info(
                    "Queued contradiction: %s -> %s",
                    target.content[:60], content[:60],
                )
            elif target:
                # Legacy path (no DB): auto-apply contradiction
                update_node(
                    target.id,
                    {"status": "contradicted", "contradiction_note": contradiction_note},
                    knowledge_path,
                )
                result.nodes_contradicted += 1
                cluster_valid_from = _cluster_valid_from(cluster)
                now = datetime.now(timezone.utc).isoformat()
                new_node = KnowledgeNode.create(
                    content=content,
                    category=category,
                    source_sessions=cluster_sessions,
                    confidence=compute_confidence(len(cluster_sessions)),
                    tags=tags,
                )
                new_node.valid_from = cluster_valid_from or now
                update_node(target.id, {"superseded_by": new_node.id}, knowledge_path)
                append_node(new_node, knowledge_path)
                result.nodes_created += 1
            else:
                # Target not found — create as new node instead
                action = "create"
            if decision_log_path and target:
                _log_dedup_decision(
                    decision_log_path,
                    action="contradict-queued" if db else "contradict",
                    candidate_content=content,
                    candidate_category=category,
                    existing_id=existing_id,
                    existing_content=target.content,
                    source="llm",
                    session_ids=cluster_sessions,
                    contradiction_note=contradiction_note,
                )
            if action != "create":
                continue  # Skip to next node (queued or legacy-applied)

        if action == "create":
            # Dedup: if content is very similar to an existing node,
            # auto-convert to corroborate instead of creating a duplicate.
            # Two signals: (1) keyword Jaccard, (2) embedding cosine.
            # Thresholds adapt to content type: personal content uses
            # higher thresholds (preserve nuance), code uses lower
            # thresholds (aggressively merge generic tool output).
            jaccard_thresh, cosine_thresh = _get_dedup_thresholds(content_profile)
            new_kw = _extract_keywords(content)
            best_match = None
            best_sim = 0.0
            best_method = "jaccard"
            all_sims: list[tuple[float, KnowledgeNode]] = []
            for existing in existing_nodes:
                sim = _jaccard(new_kw, _extract_keywords(existing.content))
                if sim > best_sim:
                    best_sim = sim
                    best_match = existing
                    best_method = "jaccard"
                if sim > 0:
                    all_sims.append((sim, existing))

            # If Jaccard didn't trigger, try embedding cosine similarity
            if best_sim < jaccard_thresh and existing_nodes:
                emb_match, emb_sim = _inline_embedding_dedup(
                    content, existing_nodes, threshold=cosine_thresh,
                )
                if emb_match and emb_sim > best_sim:
                    best_match = emb_match
                    best_sim = emb_sim
                    best_method = "cosine"

            if best_match and best_sim >= jaccard_thresh:
                logger.info(
                    "Auto-corroborate (%s=%.2f): %s",
                    best_method, best_sim, content[:80],
                )
                new_sources = list(set(best_match.source_sessions + cluster_sessions))
                new_confidence = compute_confidence(len(new_sources))
                update_node(
                    best_match.id,
                    {"source_sessions": new_sources, "confidence": new_confidence},
                    knowledge_path,
                )
                result.nodes_corroborated += 1
                if decision_log_path:
                    _log_dedup_decision(
                        decision_log_path,
                        action="auto-corroborate",
                        candidate_content=content,
                        candidate_category=category,
                        existing_id=best_match.id,
                        existing_content=best_match.content,
                        similarity_score=best_sim,
                        source=f"auto-{best_method}",
                        session_ids=cluster_sessions,
                    )
                continue

            confidence = raw_node.get("confidence", 0.5)
            if not isinstance(confidence, (int, float)):
                confidence = 0.5
            new_node = KnowledgeNode.create(
                content=content,
                category=category,
                source_sessions=cluster_sessions,
                confidence=min(1.0, max(0.0, confidence)),
                tags=tags,
                source_turns=source_turns,
            )
            # Use LLM-extracted temporal bounds if provided, else default.
            # Validate dates — LLMs can hallucinate non-ISO formats.
            if _env_flag("SYNAPT_DISABLE_TEMPORAL_EXTRACTION"):
                llm_valid_from = None
                llm_valid_until = None
            else:
                llm_valid_from = _validate_iso_date(raw_node.get("valid_from"))
                llm_valid_until = _validate_iso_date(raw_node.get("valid_until"))
            cluster_valid_from = _cluster_valid_from(cluster)
            new_node.valid_from = (
                llm_valid_from
                or cluster_valid_from
                or datetime.now(timezone.utc).isoformat()
            )
            if llm_valid_until:
                new_node.valid_until = llm_valid_until
            append_node(new_node, knowledge_path)
            existing_nodes.append(new_node)  # Track for intra-batch dedup
            result.nodes_created += 1
            if decision_log_path:
                neg_pairs = []
                if all_sims:
                    all_sims.sort(key=lambda x: x[0], reverse=True)
                    for sim_score, node in all_sims[:3]:
                        neg_pairs.append({
                            "existing_id": node.id,
                            "existing_content": node.content,
                            "similarity_score": round(sim_score, 4),
                        })
                _log_dedup_decision(
                    decision_log_path,
                    action="create",
                    candidate_content=content,
                    candidate_category=category,
                    source="llm",
                    session_ids=cluster_sessions,
                    negative_pairs=neg_pairs if neg_pairs else None,
                )

    return result


def score_cluster_chunks(cluster: list[JournalEntry]):
    """Score a cluster of journal entries via the active scoring strategy.

    Per config#332 consolidation-primary locus + config#339 Pattern 4 ratification:
    consolidation is the primary site for substrate-reshape scoring. This helper
    exposes the `ChunkScoringStrategy` seam at the canonical consolidation
    scoring point.

    OSS default (RecencyScoring): scores by temporal position (older → newer).
    Premium plugins may register alternative strategies via the registry seam;
    OSS does not name premium-side implementations.

    Backward-compat: with no plugin activated, returned scores reflect linear
    recency ramp; existing consolidation logic that uses `entry.timestamp`
    ordering produces equivalent ordering. This helper is the integration seam
    for plugin-registered strategies, not a replacement for existing
    temporal-sort logic.

    Args:
        cluster: journal entries to score (any order).

    Returns:
        List of ScoredChunks in cluster-by-timestamp order (oldest → newest).
    """
    from synapt.recall.scoring import (
        ScoringInput,
        get_active_strategy,
        score_with_validation,
    )

    if not cluster:
        return []

    ordered = sorted(cluster, key=lambda e: e.timestamp or "")
    inputs = [
        ScoringInput(
            content=entry.focus or "",
            position=i,
            metadata={
                "timestamp": entry.timestamp,
                "session_id": entry.session_id,
                "done": list(entry.done),
                "decisions": list(entry.decisions),
                "next_steps": list(entry.next_steps),
            },
        )
        for i, entry in enumerate(ordered)
    ]
    return score_with_validation(get_active_strategy(), inputs)


# ---------------------------------------------------------------------------
# recall#865 — wire extract into recall (consolidation slot, SYNAPT_USE_EXTRACT)
#
# Spec: config/design/move-1-extract-into-recall-2026-07-12.md
# Opt-in via SYNAPT_USE_EXTRACT=1. Legacy path (above) is untouched when off.
# ---------------------------------------------------------------------------

EXTRACTION_CAPABILITIES = ["facts", "decisions", "temporal_refs"]


def _extract_source_id(text: str) -> str:
    """Content-derived hash so the extraction packet has a real address.

    Not automatic from extract itself — extract only hashes source text for
    the optional artifact-bundle wrapper, not the packet. Setting source_id
    here gives the packet the "addressable-unit" property from the
    2026-05-08 intelligence-packet reframe.

    source_id is a generic identifier field in extract's schema (a plain
    optional string — no built-in content-hash semantic; verified against
    schema.py/validate.py) so this doesn't stomp an existing extract
    meaning. Forward-compat: if extract later adds a native packet-level
    content_hash field, migrate to that instead of overloading source_id.
    recall's own provenance (source_sessions / source_turns on
    KnowledgeNode) is tracked separately from this field.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _map_extraction_category(category: str | None) -> str:
    """Map extract's free-form fact.category into recall's VALID_CATEGORIES,
    defaulting to "fact" on no match."""
    if category and category in VALID_CATEGORIES:
        return category
    return "fact"


def _earliest_temporal_ref_date(temporal_refs: list[dict]) -> str | None:
    """Earliest parseable resolved date across *temporal_refs*, or None.

    temporal_refs isn't per-fact-linked without the evidence_anchoring
    capability (more prompt surface than this slot requests), so this sets
    valid_from for every node produced from the same packet — the same
    "earliest wins" shape as _cluster_valid_from() uses for journal entries.
    """
    earliest: tuple[datetime, str] | None = None
    for ref in temporal_refs:
        if not isinstance(ref, dict):
            continue
        resolved = ref.get("resolved")
        validated = _validate_iso_date(resolved)
        if not validated:
            continue
        try:
            dt = datetime.fromisoformat(validated.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        if earliest is None or dt < earliest[0]:
            earliest = (dt, validated)
    return earliest[1] if earliest is not None else None


def _knowledge_nodes_from_extraction(
    packet: dict,
    cluster_sessions: list[str],
) -> list[KnowledgeNode]:
    """Map a validated SynaptExtraction packet's facts/decisions into
    KnowledgeNodes.

    entities are NOT materialized as nodes in this slot — they're
    structural (name/type/aliases), not durable prose. Confidence reuses
    recall's existing compute_confidence() rather than extract's per-item
    signals.confidence (keeps the requested capability surface minimal).
    Dedup/corroborate/contradict against existing knowledge is handled by
    the existing post-consolidation pipeline (dedup_knowledge_nodes(),
    resolve_source_turns(), resolve_source_offsets()) uniformly for nodes
    from either path — not duplicated here.
    """
    valid_from = (
        _earliest_temporal_ref_date(packet.get("temporal_refs") or [])
        or datetime.now(timezone.utc).isoformat()
    )
    confidence = compute_confidence(len(cluster_sessions))
    nodes: list[KnowledgeNode] = []

    for fact in packet.get("facts") or []:
        text = fact.get("text") if isinstance(fact, dict) else None
        if not text:
            continue
        node = KnowledgeNode.create(
            content=text,
            category=_map_extraction_category(fact.get("category")),
            source_sessions=cluster_sessions,
            confidence=confidence,
        )
        node.valid_from = valid_from
        nodes.append(node)

    for decision in packet.get("decisions") or []:
        text = decision.get("text") if isinstance(decision, dict) else None
        if not text:
            continue
        node = KnowledgeNode.create(
            content=text,
            category="decision",
            source_sessions=cluster_sessions,
            confidence=confidence,
        )
        node.valid_from = valid_from
        nodes.append(node)

    return nodes


def _build_extraction_packet(
    text: str,
    client,
    model: str,
    *,
    adapter_path: str = "",
    max_tokens: int = MIN_RESPONSE_TOKENS,
) -> dict | None:
    """Run extract Stage 1 (via *client*, synchronously) → finalize →
    validate.

    Bypasses extract.py's async extract() orchestrator — consolidate.py is
    fully sync and MLXClient.chat() is a blocking call, so this calls
    create_extraction_builder().build() for the prompt/schema, the existing
    MLX/router client for inference, and finalize_extraction() directly,
    all synchronously.

    Returns the finalized, validated SynaptExtraction packet (dict), or
    None if the LLM response is unparseable or fails schema validation
    (fail-closed — mirrors _process_cluster's unparseable-response handling
    on the legacy path; no fallback to the legacy path mid-flag-on).
    """
    built = create_extraction_builder(
        text=text,
        capabilities=list(EXTRACTION_CAPABILITIES),
    ).build()

    try:
        response = client.chat(
            model=model,
            messages=[Message(role="user", content=built["prompt"])],
            temperature=0.1,
            adapter_path=adapter_path or None,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        logger.warning("Extraction Stage-1 inference failed: %s", exc)
        return None

    parsed = _parse_llm_response(response)
    if not parsed:
        logger.warning(
            "Unparseable extraction Stage-1 response (%d chars): %.300s",
            len(response), response,
        )
        return None

    # extract's finalized schema unconditionally requires entities/goals/
    # themes arrays even though the Stage-1 request schema only requires
    # the capabilities we asked for — backfill so validation doesn't fail
    # on containers we deliberately didn't request. Also defensively
    # backfill the requested arrays in case the model dropped one.
    for key in ("entities", "goals", "themes", *EXTRACTION_CAPABILITIES):
        parsed.setdefault(key, [])

    context = FinalizeContext(
        produced_by=f"mlx://{model}",
        source_id=_extract_source_id(text),
        capabilities_hint=built["capabilities"],
    )
    finalized = finalize_extraction(parsed, context)
    if not finalized.validation.valid:
        logger.warning(
            "Extraction packet failed schema validation (%d error(s)): %s",
            len(finalized.validation.errors),
            "; ".join(f"{e.path}: {e.message}" for e in finalized.validation.errors[:5]),
        )
        return None

    return finalized.extraction


def consolidate(
    project_dir: Path | None = None,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
    force: bool = False,
    min_entries: int = 3,
    adapter_path: str = "",
    content_profile=None,
    explicit_journal_path: Path | None = None,
    explicit_knowledge_path: Path | None = None,
) -> ConsolidationResult:
    """Run memory consolidation: extract knowledge from journal entries.

    Args:
        project_dir: Project root. Default: cwd.
        model: MLX model for knowledge extraction.
        dry_run: Show what would happen without making changes.
        force: Ignore last_consolidation_ts, reprocess all entries.
        min_entries: Minimum enriched journal entries to trigger consolidation.
        adapter_path: Optional LoRA adapter for knowledge extraction.
        explicit_journal_path: Override journal path (for temp dirs / eval adapters).
        explicit_knowledge_path: Override knowledge path (for temp dirs / eval adapters).

    Returns:
        Summary of what was created/corroborated/contradicted.
    """
    project_dir = (project_dir or Path.cwd()).resolve()
    journal_path = explicit_journal_path or _journal_path(project_dir)
    kn_path = explicit_knowledge_path or _knowledge_path(project_dir)
    result = ConsolidationResult()

    if not journal_path.exists():
        return result

    # Auto-detect content profile from transcript chunks if not provided
    if content_profile is None:
        try:
            from synapt.recall.content_profile import detect_content_profile
            idx_dir = project_index_dir(project_dir)
            from synapt.recall.storage import RecallDB
            db_path = idx_dir / "recall.db"
            if db_path.exists():
                _db = RecallDB(db_path)
                from synapt.recall.core import parse_transcript
                # Sample chunks from the DB for classification
                chunk_texts = _db.sample_chunk_texts(limit=100)
                if chunk_texts:
                    # Create lightweight objects with .text attribute
                    class _TextHolder:
                        def __init__(self, t): self.text = t
                    content_profile = detect_content_profile(
                        [_TextHolder(t) for t in chunk_texts]
                    )
                    logger.info("Auto-detected content profile: %s", content_profile.content_type)
                _db.close()
        except Exception:
            logger.debug("Content profile auto-detection failed", exc_info=True)

    # Read and dedup journal entries
    raw_entries = _read_all_entries(journal_path)
    entries = _dedup_entries(raw_entries)

    # Filter to enriched entries with rich content
    rich_entries = [
        e for e in entries
        if e.has_rich_content()
    ]

    # Filter by last consolidation timestamp (unless --force)
    if not force:
        last_ts = _get_last_consolidation_ts(project_dir)
        if last_ts:
            rich_entries = [e for e in rich_entries if e.timestamp > last_ts]

    if len(rich_entries) < min_entries:
        logger.info(
            "Only %d enriched entries since last consolidation (need %d)",
            len(rich_entries), min_entries,
        )
        if dry_run:
            print(
                f"  Not enough enriched entries: {len(rich_entries)} "
                f"(need at least {min_entries})"
            )
        return result

    result.entries_processed = len(rich_entries)

    # Cluster related entries
    clusters = cluster_journal_entries(rich_entries)
    result.clusters_found = len(clusters)

    if dry_run:
        print(f"  Entries to process: {len(rich_entries)}")
        print(f"  Clusters found: {len(clusters)}")
        for i, cluster in enumerate(clusters):
            sessions = [e.session_id[:8] for e in cluster if e.session_id]
            foci = [e.focus[:60] for e in cluster if e.focus]
            print(f"  Cluster {i+1}: {len(cluster)} entries — sessions: {', '.join(sessions)}")
            for f in foci:
                print(f"    - {f}")
        return result

    # Load existing knowledge for context
    existing_nodes = read_nodes(kn_path, status="active")
    decision_path = _dedup_decisions_path(project_dir)

    # Open DB for queuing contradictions (Phase 8b)
    db = None
    index_dir = project_index_dir(project_dir)
    db_path = index_dir / "recall.db"
    if db_path.exists():
        try:
            from synapt.recall.storage import RecallDB
            db = RecallDB(db_path)
        except Exception:
            pass  # Fall back to legacy auto-apply

    # Load cluster-level LLM response cache
    data_dir = project_data_dir(project_dir)
    cache_path = data_dir / "consolidation_cache.jsonl"
    failures_path = data_dir / "consolidation_failures.jsonl"
    response_cache = _load_response_cache(cache_path)

    # Get model client via router (MLX → Modal → Ollama)
    client = None

    def _process_cluster(cluster: list[JournalEntry]) -> bool:
        """Process a single cluster. Returns True if successful."""
        nonlocal client

        use_extract = _env_flag("SYNAPT_USE_EXTRACT")

        cache_key = _cluster_cache_key(cluster)
        if not use_extract:
            # recall#865: the extract path doesn't share the legacy
            # response cache — the prompt shapes differ, and reusing the
            # same cache key across paths would risk feeding a
            # legacy-shaped cached response into the extract parser (or
            # vice versa) when the flag is toggled between runs.
            cached_entry = response_cache.get(cache_key)
            if cached_entry:
                # Response was already applied on a previous run — the side
                # effects (append_node / update_node) are on disk.  Re-applying
                # would create duplicate nodes and double-count corroborations.
                logger.debug("Cache hit for cluster %s — skipping", cache_key)
                return True

        if client is None:
            from synapt.recall._model_router import get_client, RecallTask
            client = get_client(RecallTask.CONSOLIDATE, max_tokens=MIN_RESPONSE_TOKENS)
            if client is None:
                if not _MLX_AVAILABLE:
                    logger.error("No model backend available for consolidation")
                    return False
                from synapt._models.mlx_client import MLXClient, MLXOptions
                client = MLXClient(MLXOptions(max_tokens=MIN_RESPONSE_TOKENS))

        if use_extract:
            cluster_text = _format_journal_cluster(cluster)
            packet = _build_extraction_packet(
                cluster_text, client, model, adapter_path=adapter_path,
            )
            if packet is None:
                return False
            cluster_sessions = [e.session_id for e in cluster if e.session_id]
            new_nodes = _knowledge_nodes_from_extraction(packet, cluster_sessions)
            for node in new_nodes:
                append_node(node, kn_path)
                existing_nodes.append(node)
            result.nodes_created += len(new_nodes)
            return True

        prompt = _build_consolidation_prompt(
            cluster, existing_nodes, project_dir,
            adapter_path=adapter_path,
        )
        response_budget = _estimate_response_budget(prompt)
        try:
            response = client.chat(
                model=model,
                messages=[Message(role="user", content=prompt)],
                temperature=0.1,
                adapter_path=adapter_path or None,
                max_tokens=response_budget,
            )
        except Exception as exc:
            logger.warning("Inference failed for cluster: %s", exc)
            return False

        parsed = _parse_llm_response(response)
        if not parsed:
            logger.warning(
                "Unparseable LLM response (%d chars): %.300s",
                len(response), response,
            )
            # Save failure for diagnostics (full prompt + response)
            try:
                from synapt.recall._filelock import lock_exclusive, unlock
                with open(failures_path, "a", encoding="utf-8") as ff:
                    lock_exclusive(ff)
                    try:
                        ff.write(json.dumps({
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "key": cache_key,
                            "prompt": prompt,
                            "response": response,
                            "cluster_sessions": [e.session_id for e in cluster],
                        }) + "\n")
                    finally:
                        unlock(ff)
            except OSError:
                pass
            return False

        cluster_result = _apply_consolidation_result(
            parsed, existing_nodes, cluster, kn_path,
            decision_log_path=decision_path,
            db=db,
            content_profile=content_profile,
        )
        result.nodes_created += cluster_result.nodes_created
        result.nodes_corroborated += cluster_result.nodes_corroborated
        result.nodes_contradicted += cluster_result.nodes_contradicted

        # Cache successful response + prompt for future runs / adapter training
        _save_cached_response(cache_path, cache_key, response, prompt)
        response_cache[cache_key] = {"response": response, "prompt": prompt}

        return True

    try:
        n_clusters = len(clusters)
        for ci, cluster in enumerate(clusters):
            logger.info(
                "Consolidating cluster %d/%d (%d entries)",
                ci + 1, n_clusters, len(cluster),
            )
            if _process_cluster(cluster):
                # Reload existing nodes for subsequent clusters
                existing_nodes = read_nodes(kn_path, status="active")
                logger.info(
                    "Cluster %d/%d done: %d nodes created, %d corroborated",
                    ci + 1, n_clusters,
                    result.nodes_created, result.nodes_corroborated,
                )
                continue

            # Retry: split failed cluster in half and process each sub-cluster
            if len(cluster) >= 3:
                mid = len(cluster) // 2
                sorted_entries = sorted(cluster, key=lambda e: e.timestamp or "")
                halves = [sorted_entries[:mid], sorted_entries[mid:]]
                for half in halves:
                    if len(half) < 2:
                        logger.debug("Skipped sub-cluster with %d entry", len(half))
                    elif _process_cluster(half):
                        existing_nodes = read_nodes(kn_path, status="active")
                    else:
                        logger.warning(
                            "Failed to parse LLM response for sub-cluster (%d entries)",
                            len(half),
                        )
            else:
                logger.warning(
                    "Failed to parse LLM response for cluster (%d entries)",
                    len(cluster),
                )

        # Sync knowledge.jsonl → SQLite when nodes were modified OR when
        # the knowledge file has nodes that may not be in SQLite yet
        # (e.g. from a prior run that wrote to JSONL but crashed before sync).
        if result.nodes_created or result.nodes_corroborated or result.nodes_contradicted:
            _set_last_consolidation_ts(project_dir)

            # Post-consolidation dedup — merges near-duplicates that the
            # inline Jaccard check missed (e.g. semantic duplicates with
            # different wording, or nodes created within the same batch).
            # Uses content-type-aware thresholds.
            _jt, _ct = _get_dedup_thresholds(content_profile)
            merged = dedup_knowledge_nodes(
                threshold=_jt, project_dir=project_dir,
                embedding_threshold=_ct,
            )
            if merged:
                logger.info(
                    "Post-consolidation dedup merged %d duplicate(s)", merged,
                )
                result.nodes_deduped += merged

            _sync_knowledge_to_db(project_dir, kn_path)
            # Resolve source_turns and source_offsets for any new/updated nodes
            resolved = resolve_source_turns(project_dir)
            if resolved:
                logger.info("Resolved source_turns for %d knowledge nodes", resolved)
            resolved_offsets = resolve_source_offsets(project_dir)
            if resolved_offsets:
                logger.info("Resolved source_offsets for %d knowledge nodes", resolved_offsets)
            # Single final sync after all resolvers have written their updates
            if resolved or resolved_offsets:
                _sync_knowledge_to_db(project_dir, kn_path)

            # Collection extraction: find entity groups across sessions
            if not _env_flag("SYNAPT_DISABLE_ENTITY_COLLECTION"):
                collections_created = extract_collections(
                    project_dir, model=model, adapter_path=adapter_path,
                )
                if collections_created:
                    result.nodes_created += collections_created
        elif clusters and kn_path.exists() and kn_path.stat().st_size > 0:
            _sync_knowledge_to_db(project_dir, kn_path)
    finally:
        if db is not None:
            db.close()

    return result


def extract_collections(
    project_dir: Path | None = None,
    model: str = DEFAULT_MODEL,
    adapter_path: str = "",
) -> int:
    """Extract entity-collection knowledge nodes from existing knowledge.

    Runs as a second pass after standard consolidation. Scans all active
    knowledge nodes to find groups of 3+ similar entities across sessions,
    then creates "collection" category nodes that enumerate them.

    Returns number of collection nodes created.
    """
    if _env_flag("SYNAPT_DISABLE_ENTITY_COLLECTION"):
        return 0

    project_dir = (project_dir or Path.cwd()).resolve()
    kn_path = _knowledge_path(project_dir)

    if not kn_path.exists():
        return 0

    existing_nodes = read_nodes(kn_path, status="active")
    if len(existing_nodes) < 5:
        # Not enough knowledge to find meaningful collections
        return 0

    # Skip if we already have collection nodes and no new non-collection nodes
    collection_nodes = [n for n in existing_nodes if n.category == "collection"]
    non_collection = [n for n in existing_nodes if n.category != "collection"]
    if not non_collection:
        return 0

    # Format existing knowledge for the prompt
    lines = []
    for node in non_collection:
        sessions = ", ".join(node.source_sessions[:5]) if node.source_sessions else "?"
        lines.append(f"- [{node.category}] {node.content} (sessions: {sessions})")
    existing_text = "\n".join(lines[:100])  # Cap to avoid context overflow

    # Also include existing collection nodes so the model can corroborate/update
    if collection_nodes:
        existing_text += "\n\n## Existing Collections\n"
        for cn in collection_nodes:
            existing_text += f"- [id={cn.id}] {cn.content}\n"

    prompt = COLLECTION_EXTRACTION_PROMPT.format(existing_knowledge=existing_text)

    # Get model client
    from synapt.recall._model_router import get_client, RecallTask
    client = get_client(RecallTask.CONSOLIDATE, max_tokens=MIN_RESPONSE_TOKENS)
    if client is None:
        if not _MLX_AVAILABLE:
            logger.warning("No model backend available for collection extraction")
            return 0
        from synapt._models.mlx_client import MLXClient, MLXOptions
        client = MLXClient(MLXOptions(max_tokens=MIN_RESPONSE_TOKENS))

    try:
        response = client.chat(
            model=model,
            messages=[Message(role="user", content=prompt)],
            temperature=0.1,
            adapter_path=adapter_path or None,
            max_tokens=MIN_RESPONSE_TOKENS,
        )
    except Exception as exc:
        logger.warning("Collection extraction inference failed: %s", exc)
        return 0

    parsed = _parse_llm_response(response)
    if not parsed:
        logger.debug("No collections extracted (unparseable response)")
        return 0

    # Apply results — only create/corroborate, use the standard apply function
    dummy_cluster: list[JournalEntry] = []  # No journal entries for collection pass
    collection_result = _apply_consolidation_result(
        parsed, existing_nodes, dummy_cluster, kn_path,
    )

    if collection_result.nodes_created:
        logger.info("Extracted %d collection node(s)", collection_result.nodes_created)
        _sync_knowledge_to_db(project_dir, kn_path)

    return collection_result.nodes_created


def _get_last_consolidation_ts(project_dir: Path) -> str:
    """Read last consolidation timestamp from metadata."""
    index_dir = project_index_dir(project_dir)
    db_path = index_dir / "recall.db"
    if not db_path.exists():
        return ""
    try:
        from synapt.recall.storage import RecallDB
        db = RecallDB(db_path)
        ts = db.get_metadata("last_consolidation_ts") or ""
        db.close()
        return ts
    except Exception:
        return ""


def _set_last_consolidation_ts(project_dir: Path) -> None:
    """Write current timestamp as last consolidation timestamp."""
    index_dir = project_index_dir(project_dir)
    db_path = index_dir / "recall.db"
    if not db_path.exists():
        return
    try:
        from synapt.recall.storage import RecallDB
        db = RecallDB(db_path)
        db.set_metadata(
            "last_consolidation_ts",
            datetime.now(timezone.utc).isoformat(),
        )
        db.close()
    except Exception:
        pass


def _sync_knowledge_to_db(project_dir: Path, kn_path: Path) -> None:
    """Sync knowledge.jsonl nodes into SQLite for FTS search."""
    index_dir = project_index_dir(project_dir)
    db_path = index_dir / "recall.db"
    if not db_path.exists():
        return
    try:
        from synapt.recall.storage import RecallDB
        nodes = read_nodes(kn_path)
        if not nodes:
            return
        db = RecallDB(db_path)
        db.save_knowledge_nodes([n.to_dict() for n in nodes])
        db.close()
    except Exception as exc:
        logger.warning("Failed to sync knowledge to DB: %s", exc)


def _load_chunks_for_resolve(
    project_dir: Path | None = None,
    *,
    include_text: bool = False,
) -> tuple[Path, dict, list[KnowledgeNode], "RecallDB"] | None:  # noqa: F821
    """Load chunks and active nodes for source resolution.

    Returns (kn_path, session_chunks, nodes, db) or None if data unavailable.

    session_chunks maps session_id -> [(turn_index, token_set)] or
    session_id -> [(turn_index, full_text, token_set)] if include_text=True.

    Only loads chunks from sessions referenced by active nodes.
    """
    project_dir = (project_dir or Path.cwd()).resolve()
    kn_path = _knowledge_path(project_dir)
    if not kn_path.exists():
        return None

    index_dir = project_index_dir(project_dir)
    db_path = index_dir / "recall.db"
    if not db_path.exists():
        return None

    from synapt.recall.storage import RecallDB

    db = RecallDB(db_path)

    # Load active knowledge nodes
    nodes = read_nodes(kn_path, status="active")
    if not nodes:
        db.close()
        return None

    # Collect all source_sessions referenced by active nodes
    needed_sessions: set[str] = set()
    for node in nodes:
        needed_sessions.update(node.source_sessions)

    if not needed_sessions:
        db.close()
        return None

    # Filter chunks to only relevant sessions
    placeholders = ",".join("?" for _ in needed_sessions)
    rows = db._conn.execute(
        f"SELECT session_id, turn_index, user_text, assistant_text, transcript_path "
        f"FROM chunks WHERE session_id IN ({placeholders})",
        list(needed_sessions),
    ).fetchall()
    if not rows:
        db.close()
        return None

    # Build session_id → chunk list
    session_chunks: dict = {}
    for r in rows:
        sid = r["session_id"]
        tidx = r["turn_index"]
        text = (r["user_text"] or "") + " " + (r["assistant_text"] or "")
        toks = _extract_keywords(text)
        tpath = r["transcript_path"] or ""
        if include_text:
            session_chunks.setdefault(sid, []).append((tidx, text, toks, tpath))
        else:
            session_chunks.setdefault(sid, []).append((tidx, toks))

    return kn_path, session_chunks, nodes, db


def resolve_source_turns(
    project_dir: Path | None = None,
    *,
    max_turns_per_node: int = 3,
    min_overlap: int = 3,
) -> int:
    """Resolve source_turns for knowledge nodes by matching against chunks.

    For each active knowledge node with empty source_turns, find transcript
    chunks from its source_sessions whose text has high token overlap with
    the node content.  Stores the top-N turn references as source_turns.

    This enables precise O(1) source expansion in retrieval instead of the
    broad keyword fallback that scans all chunks in source sessions.

    Returns the number of nodes updated.
    """
    loaded = _load_chunks_for_resolve(project_dir, include_text=False)
    if loaded is None:
        return 0

    kn_path, session_chunks, nodes, db = loaded
    pending_updates: dict[str, dict] = {}

    try:
        for node in nodes:
            if node.source_turns:
                continue  # Already has source_turns

            if not node.source_sessions:
                continue

            node_toks = _extract_keywords(node.content)
            if len(node_toks) < 2:
                continue

            # Score every chunk in this node's source sessions
            candidates: list[tuple[str, int, int]] = []  # (session_id, turn_idx, overlap)
            for sid in node.source_sessions:
                for tidx, chunk_toks in session_chunks.get(sid, []):
                    overlap = len(node_toks & chunk_toks)
                    if overlap >= min_overlap:
                        candidates.append((sid, tidx, overlap))

            if not candidates:
                continue

            # Take top-N by overlap
            candidates.sort(key=lambda x: x[2], reverse=True)
            source_turns = [
                f"{sid}:{tidx}" for sid, tidx, _ in candidates[:max_turns_per_node]
            ]

            pending_updates[node.id] = {"source_turns": source_turns}
    finally:
        db.close()

    if pending_updates:
        batch_update_nodes(pending_updates, kn_path)

    return len(pending_updates)


def _find_best_span(node_text: str, chunk_text: str, margin: int = 30) -> tuple[int, int] | None:
    """Find the character span in chunk_text that best covers node_text content.

    Splits chunk_text into sentences, scores each by token overlap with
    node_text, then returns (begin, end) covering the best contiguous
    sentence window.  Adds ``margin`` chars of context on each side.
    """
    if not chunk_text or not node_text:
        return None

    # Split into sentences (period/question/exclamation followed by space or end)
    sentence_spans: list[tuple[int, int, str]] = []
    for m in re.finditer(r'[^.!?]*[.!?]+(?:\s|$)|[^.!?]+$', chunk_text):
        sentence_spans.append((m.start(), m.end(), m.group()))

    if not sentence_spans:
        return None

    node_toks = _extract_keywords(node_text)
    if not node_toks:
        return None

    # Score each sentence
    scored: list[tuple[int, int, int]] = []  # (begin, end, overlap)
    for begin, end, sent in sentence_spans:
        sent_toks = _extract_keywords(sent)
        overlap = len(node_toks & sent_toks)
        scored.append((begin, end, overlap))

    # Find best contiguous window of 1-3 sentences
    best_score = 0
    best_begin = 0
    best_end = 0
    for window in range(1, min(4, len(scored) + 1)):
        for i in range(len(scored) - window + 1):
            total = sum(scored[j][2] for j in range(i, i + window))
            if total > best_score:
                best_score = total
                best_begin = scored[i][0]
                best_end = scored[i + window - 1][1]

    if best_score < 1:
        return None

    # Add margin for context
    begin = max(0, best_begin - margin)
    end = min(len(chunk_text), best_end + margin)
    return (begin, end)


def resolve_source_offsets(
    project_dir: Path | None = None,
    *,
    max_offsets_per_node: int = 3,
    min_overlap: int = 2,
) -> int:
    """Resolve source_offsets for knowledge nodes by finding sentence spans.

    For each active knowledge node, finds the best-matching sentence spans
    within transcript chunks from its source_sessions.  Stores character
    offsets (begin, end) so retrieval can extract precise snippets instead
    of formatting entire turns.

    When a node already has ``source_turns`` (from ``resolve_source_turns``),
    those turns are used as direct candidates — skipping the token-overlap
    scan entirely.  This improves coverage for short or abstract nodes that
    would otherwise fail the overlap threshold.

    Returns the number of nodes updated.
    """
    loaded = _load_chunks_for_resolve(project_dir, include_text=True)
    if loaded is None:
        return 0

    kn_path, session_chunks, nodes, db = loaded
    pending_updates: dict[str, dict] = {}

    # Build turn lookups: (session_id, turn_index) → full text and file path
    _turn_text: dict[tuple[str, int], str] = {}
    _turn_file: dict[tuple[str, int], str] = {}
    for sid, chunks_list in session_chunks.items():
        for tidx, text, _toks, tpath in chunks_list:
            _turn_text[(sid, tidx)] = text
            if tpath:
                _turn_file[(sid, tidx)] = tpath

    try:
        for node in nodes:
            if node.source_offsets:
                continue  # Already resolved

            # Fast path: use source_turns as direct candidates when available.
            # source_turns are already the best-matching turns from
            # resolve_source_turns(), so we skip the token overlap scan.
            if node.source_turns:
                offsets: list[dict] = []
                for turn_ref in node.source_turns[:max_offsets_per_node]:
                    # Parse "session_id:turn_index" format
                    parts = turn_ref.rsplit(":", 1)
                    if len(parts) != 2:
                        continue
                    sid, tidx_str = parts
                    try:
                        tidx = int(tidx_str)
                    except ValueError:
                        continue
                    text = _turn_text.get((sid, tidx), "")
                    if not text:
                        continue
                    span = _find_best_span(node.content, text)
                    if span:
                        entry: dict = {
                            "s": sid, "t": tidx,
                            "b": span[0], "e": span[1],
                        }
                        fpath = _turn_file.get((sid, tidx), "")
                        if fpath:
                            entry["f"] = fpath
                        offsets.append(entry)
                if offsets:
                    pending_updates[node.id] = {"source_offsets": offsets}
                    continue

            # Fallback: scan source_sessions by token overlap
            if not node.source_sessions:
                continue

            node_toks = _extract_keywords(node.content)
            if len(node_toks) < 2:
                continue

            candidates: list[tuple[str, int, str, int, str]] = []
            for sid in node.source_sessions:
                for tidx, text, chunk_toks, tpath in session_chunks.get(sid, []):
                    overlap = len(node_toks & chunk_toks)
                    if overlap >= min_overlap:
                        candidates.append((sid, tidx, text, overlap, tpath))

            if not candidates:
                continue

            candidates.sort(key=lambda x: x[3], reverse=True)

            offsets = []
            for sid, tidx, text, _, tpath in candidates[:max_offsets_per_node]:
                span = _find_best_span(node.content, text)
                if span:
                    entry = {
                        "s": sid, "t": tidx,
                        "b": span[0], "e": span[1],
                    }
                    if tpath:
                        entry["f"] = tpath
                    offsets.append(entry)

            if offsets:
                pending_updates[node.id] = {"source_offsets": offsets}
    finally:
        db.close()

    if pending_updates:
        batch_update_nodes(pending_updates, kn_path)

    return len(pending_updates)
