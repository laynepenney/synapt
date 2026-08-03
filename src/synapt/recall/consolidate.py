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

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
from collections import Counter
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
    # Fix B contested-memory-lifecycle reframe (internal design spec, section 10.5):
    # distinct from nodes_contradicted (the
    # general contradict-action-item path) so the 0.17.0 dogfood fruit is countable cleanly --
    # one increment per contested PAIR (two nodes touched, one queue entry), not per node.
    nodes_contested: int = 0
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


# Session/command BOOKKEEPING dressed up as durable facts. Distinct from
# _GARBLED_CONTENT_PATTERNS above, which catches raw LLM *structural* metadata
# (existing_id:, contradiction_note:) bleeding into the content field. This set
# catches well-formed prose whose SUBJECT is journal bookkeeping or a one-off
# command execution -- content that parses cleanly and means nothing later.
#
# _lacks_specificity does not catch these: it reads a bare date as a specificity
# SIGNAL, so a bare "Session <hex-id>, <date>" tuple passes straight through it.
#
# ---- THE AXIS THIS FILTER DISCRIMINATES ON --------------------------------
#
# NOT "does this string contain a session hash, an ISO date, or a temp path".
# Those are TOKENS, and every one of them occurs in genuinely durable facts:
#
#   "Session deadbeef is the stable OAuth replay fixture used to verify rotation"
#   "Execute /migrate on 2027-01-15 only after the backup has completed"
#   "CI must never run rm -rf /tmp/shared-cache -- concurrent jobs share it"
#
# The first version of this filter matched exactly those tokens and ate all
# three. That is the same defect _lacks_specificity has, one layer over: it
# measures specificity TOKENS rather than specificity. Widening a pattern set
# cannot fix a discriminator that is pointed at the wrong quantity, and the
# false positives a reviewer happens to find are a SAMPLE of that surface, not
# its population.
#
# The property actually being filtered is:
#
#       does this content RECORD AN OCCURRENCE, or STATE A RULE?
#
# A record of one session, one command, one deletion is worthless months later.
# A rule ABOUT sessions, commands or deletions is exactly what memory is for.
#
# So each check below is TWO-PART:
#
#   1. a SHAPE gate   -- "this content is about a session id / a slash-command
#                         invocation / an ephemeral filesystem path"
#   2. a GRAMMAR test -- "...and it reports an occurrence rather than stating
#                         something that still holds"
#
# Shape alone is never a verdict. The burden of proof sits on the REJECT side,
# because a false reject destroys real memory silently while reporting a
# cleanliness win, whereas a false keep leaves one junk node behind. Every
# element below is EARNED by a reject case pinned in
# tests/recall/test_consolidate_noise.py; speculative additions can only widen
# the over-rejection surface, so they are not made.

# Positive evidence that content states something still in force: deontic
# modality, quantification over occasions, conditional structure, hedged
# tendency, product-change framing, or explicit rule vocabulary. A hit here
# OUTRANKS every shape gate below -- this is the burden-of-proof asymmetry made
# mechanical.
#
# THE MAINTENANCE RULE, and it is the whole reason this is split in two:
#
#   * ADDING to this keep-list needs no ceremony. Its worst case is that one
#     junk node survives. Add a grammatical CLASS (a quantifier, a conditional,
#     a modality), never a string lifted from one example -- fitting to
#     individual sentences rebuilds the token filter this design replaces.
#   * ADDING a reject shape below requires a pinned reject case FIRST. Its
#     worst case is silent deletion of real memory, which nothing downstream
#     can detect.
#
# The lists are therefore deliberately lopsided: three narrow reject shapes
# against an open-ended keep vocabulary. That is the design, not an imbalance
# to tidy up.
_STANDING_RULE_RE = re.compile(
    r"(?i)(?:"
    # deontic modality -- what is permitted, required or forbidden
    r"\b(?:must|should|shall|ought\s+to|has\s+to|have\s+to|may\s+not"
    r"|cannot|can't|won't|never|always|forbidden|prohibited|require[sd]?"
    r"|reserved|expected\s+to|supposed\s+to|allowed\s+to)\b"
    # universal / negative quantification over occasions or instances
    r"|\b(?:every|each|any|anything|anyone|all|none|nothing|neither|nor"
    r"|ever|whenever|wherever)\b"
    r"|\bno\s+(?:\w+\s+){0,2}(?:may|must|should|can|will|is|are)\b"
    r"|\bno\s+permission\b"
    # plain sentential negation. A negated deletion is the OPPOSITE of a
    # deletion event ("the cache is not removed between runs"), and a filter
    # hunting deletion events that cannot see negation reads every safety rule
    # as the hazard it forbids.
    r"|\bnot\b|n't\b"
    # habitual frequency
    r"|\b(?:nightly|hourly|daily|weekly|monthly|routinely|automatically)\b"
    r"|\bper\s+\w+"
    r"|\bby\s+default\b|\bdefaults?\s+to\b"
    # conditional / restrictive structure -- a rule carrying its own scope
    r"|\b(?:if|when|once|while|unless|until|only)\b"
    r"|\bas\s+long\s+as\b|\bso\s+long\s+as\b|\bprovided\s+that\b"
    r"|\b(?:at|on)\s+(?:boot|startup|shutdown)\b"
    # explicit rule vocabulary
    r"|\b(?:convention|policy|invariant|canonical)\b"
    # HEDGED / EMPIRICAL TENDENCY. Calibrated observations are durable memory and
    # carry no modal and no quantifier at all ("in practice /tmp/build-cache is
    # removed within a second of a green build, but on the ARM runners the reaper
    # is niced so heavily it can linger"). A modality-and-quantification-only
    # override misfiles this whole class as episodic.
    r"|\bin\s+practice\b|\b(?:usually|typically|generally|normally|often|rarely"
    r"|seldom|routinely|occasionally)\b|\btends?\s+to\b|\b(?:can|may|might)\b"
    # PRODUCT-CHANGE / DEPRECATION sense of the deletion verbs. "Support for the
    # /tmp/legacy-scratch override was removed in agent 2.4" deletes a FEATURE,
    # not a directory: the grammatical object of "removed" is "support", and the
    # path only sits inside the noun phrase naming it. Shape 3 sees a path and a
    # deletion verb co-occurring and cannot see that they are unrelated -- which
    # is the same co-occurrence-without-relation defect as the token filter this
    # design replaces, one level down. Without a parser the honest fix is to
    # name the construction rather than pretend the relation is checked.
    r"|\b(?:deprecated|superseded|retired|replacement)\b"
    r"|\bno\s+longer\b|\bused\s+to\b"
    r"|\bsupport\s+for\b|\bremoved\s+(?:in|upstream)\b|\bdropped\s+in\b"
    # An interrogative is never a journal record. Bookkeeping asserts; a
    # question that survived extraction is prose someone wrote to be answered.
    r"|\?"
    r")"
)

# ---- Shape 1: journal session record ---------------------------------------
# "Session <hex-id>" where the id is a real session hash (6+ hex chars). NOT
# "Session 30" / "Session v2" / "Session 7", where "session" is a domain noun
# and the token is not a hash.
_SESSION_ID_RE = re.compile(r"(?i)\bsession\s+[0-9a-f]{6,}\b")

# The whole content is the id and a timestamp -- a two-field record with no
# predicate about the world at all: "Session a1b2c3d4, 2020-01-02".
_SESSION_RECORD_TUPLE_RE = re.compile(
    r"(?i)^\W*session\s+[0-9a-f]{6,}"
    r"(?:\s*[,;:-]?\s*(?:\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}(?::\d{2})?))*"
    r"\W*$"
)

# The predicate is about the SESSION-AS-EVENT -- when it ran, how long, what it
# was about -- rather than about anything in the world. Past tense throughout,
# because that IS the axis: a present-tense predicate ("Session deadbeef uses
# AES-GCM", "Session deadbeef is the replay fixture") states a standing property
# of a thing that merely happens to be named with a hex id.
_SESSION_EVENT_PREDICATE_RE = re.compile(
    r"(?i)(?:\b(?:"
    r"occurred|took\s+place|happened|began|ended|finished|"
    r"lasted|concluded|focused|focussed|was\s+about|was\s+spent"
    # journal-row vocabulary: the fields a session log actually carries. These
    # are safe INSIDE the id shape gate in a way they would never be outside it
    # -- "Focus: visible focus rings are required" is a durable accessibility
    # rule and reaches this predicate only if it also names a session hash.
    r"|stated\s+focus|focus\s+for|duration|spanned|elapsed"
    r"|attendance|closing\s+summary|start\s+time|end\s+time"
    r")\b"
    r"|\bfrom\s+\d{4}-\d{2}-\d{2}"
    r"|\bdated\s+\d{4}-\d{2}-\d{2})"
)


def _is_session_record(content: str) -> bool:
    """Shape 1: a journal record of one session, not a fact about a session."""
    if not _SESSION_ID_RE.search(content):
        return False
    return bool(
        _SESSION_RECORD_TUPLE_RE.match(content)
        or _SESSION_EVENT_PREDICATE_RE.search(content)
    )


# ---- Shape 2: slash-command invocation log ---------------------------------
# "Execute `/clear` command on 2020-01-02" -- a transcript echo of one command
# run once. The slash token must be a COMMAND, not a path segment, so it is
# required not to be followed by another "/" and "/tmp" is excluded outright.
_SLASH_COMMAND_RE = re.compile(r"(?i)(?:^|[\s`'\"(\[])/(?!tmp\b)\w[\w-]*(?![/\w])")
_EXECUTION_VERB_RE = re.compile(
    r"(?i)\b(?:execute[ds]?|executing|ran|run|invoked|issued|command)\b"
)
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def _is_command_invocation_log(content: str) -> bool:
    """Shape 2: a slash-command echo bound to a date, with no standing structure.

    The DATE is part of the shape gate, not the discriminator -- "Execute
    /migrate on 2027-01-15 only after the backup completes" is a future-dated
    durable instruction and is kept by the standing-rule override above.
    """
    return bool(
        _SLASH_COMMAND_RE.search(content)
        and _EXECUTION_VERB_RE.search(content)
        and _ISO_DATE_RE.search(content)
    )


# ---- Shape 3: one-off execution event against an ephemeral path ------------
# Cross-platform on purpose. recall is public and runs on POSIX, macOS and
# Windows; a /tmp-only pattern passes the byte-identical Windows event
# ("...\AppData\Local\Temp\scratch-run ... removed with Remove-Item"), which
# would make the stated class narrower than the claim made for it.
#
# The leading boundary group is load-bearing, not decoration: without it
# "/tmp/" matches as a SUBSTRING of an unrelated durable path such as
# /var/lib/nginx/tmp/client_body, which is a stable server directory and not a
# scratch tree at all. Anchoring to a real path start is the difference between
# "this content names an ephemeral location" and "these six characters occur".
_EPHEMERAL_PATH_RE = re.compile(
    r"(?i)(?:^|[\s'\"`(\[=])(?:"
    r"/tmp/\S+"                                # POSIX
    r"|/var/folders/\S+"                       # macOS per-user temp
    r"|\$TMPDIR\S*|%TEMP%\S*|%TMP%\S*"         # env-var forms
    r"|[a-z]:\\(?:[^\\\s]+\\)*Temp\\\S+"       # Windows ...\Local\Temp\...
    r")"
)
_DELETION_RE = re.compile(
    r"(?i)(?:\brm\s+-[rf]{1,2}\b|\bRemove-Item\b|\brmdir\b"
    r"|\b(?:wiped|deleted|removed|purged|cleared|unlinked)\b)"
)


def _is_ephemeral_execution_event(content: str) -> bool:
    """Shape 3: a deletion against a scratch path, with no standing structure.

    "The /tmp/build-cache directory is removed only after artifact upload
    succeeds" carries the same two tokens and is a lifecycle RULE; the
    standing-rule override above keeps it.
    """
    return bool(
        _EPHEMERAL_PATH_RE.search(content) and _DELETION_RE.search(content)
    )


# ---- THE UNIT OF JUDGMENT IS THE CLAUSE, NOT THE STRING --------------------
#
# Three review rounds failed here, and they failed the same way: each changed
# WHICH SIGNALS were consulted, and none changed THE SCOPE A SIGNAL SPEAKS OVER.
# The defect underneath all three is one error with two faces:
#
#       A SIGNAL FOUND ANYWHERE IN THE CONTENT WAS TREATED
#       AS A VERDICT ABOUT ALL OF IT.
#
#   reject side -- a bookkeeping shape in ONE clause deleted EVERY clause,
#     including clauses asserting durable facts:
#       "The /migrate command ran on 2026-08-01 and permanently upgraded the
#        tenant schema to version 12, which fixes the lock-order defect."
#     The dated command echo is real. So is the schema state and the fix. The
#     whole fact was deleted on the strength of its first clause.
#
#   keep side -- a rule word in a REPORTED clause rescued EVERY clause,
#     including pure bookkeeping:
#       "Session a1b2c3d4 focused on what should happen next."
#     "should" is MENTIONED inside what the session was about. It is not
#     asserted by anyone, and it rescued a journal row.
#
# So occurrence-versus-rule was never the wrong axis; it was evaluated at the
# wrong GRANULARITY. An occurrence can carry a durable residue, and when it does
# the residue is the memory.
#
# The classes, in the form ratified by review:
#
#   CLASS A  standing assertion -- a proposition useful or true AFTER the
#            occurrence ends, REGARDLESS OF GRAMMATICAL SUBJECT. "Session
#            deadbeef ... exposed a lock-order bug" is class A: the subject is
#            the session and the bug outlives it. Durable facts routinely
#            arrive wearing occurrence clothing.
#   CLASS B  occurrence metadata ONLY -- time, duration, topic, command
#            execution, with no durable result.
#   CLASS C  reported content -- the complement of an attitude or reporting
#            predicate. Defined SEMANTICALLY, not by a closed verb list: it is
#            what someone focused on, was about, discussed. Vocabulary there is
#            MENTIONED, never asserted, so it licenses nothing about the matrix
#            claim. The recognizer below is necessarily partial; anything it
#            cannot identify is unknown, and unknown keeps.
#
# THE DECISION RULE, and note it is asymmetric on purpose:
#
#   REJECT  only if at least one clause is POSITIVELY identified class B
#           AND every remaining clause is B or C.
#   KEEP    on any class A clause, any unknown clause, or C-only content.
#           Class C alone can never condemn.
#
# That is the reject-side burden of proof made mechanical: deleting requires
# establishing that the content carries NO durable residue anywhere, and a
# shape match on one clause is evidence about THAT CLAUSE ONLY.


# Proposition boundaries: coordination, relative clauses, participials, and
# punctuation. Segmentation without a parser is approximate, and the errors are
# deliberately biased -- OVER-splitting yields more clauses and therefore more
# chances to find a class A, which lands on the keep side. UNDER-splitting fuses
# an occurrence with its residue and would delete the residue, so it is the
# failure worth avoiding.
_PROPOSITION_SPLIT_RE = re.compile(
    r"(?:"
    r"\s*[;]\s*"                              # semicolon
    r"|\s*[.!?][\"'”’\)\]]*\s+"               # sentence punctuation + closing quotes
    r"|\s*\n+\s*"                             # newline
    r"|\s*(?:[—–―‒]|--)\s*"                   # dash family, spaced OR CLOSED
    r"|\s*,\s*(?i:(?=which\b|that\b|so\b|because\b))"   # relative / causal
    r"|\s*,\s*(?=\w+ing\b)"                   # participial: ", exposing ..."
    r"|\s+(?i:and|but)\s+"                    # coordination
    r")"
)
# THREE THINGS ABOUT THIS SET ARE DELIBERATE AND WERE EACH EARNED BY A FINDING.
#
# 1. THE COLON IS NOT HERE AT ALL. A conditional colon split was tried, and a
#    reviewer proved the condition wrong in BOTH directions with one probe each:
#      false reject -- "...concluded the migration: production rejects SHA-1"
#                      (lowercase prose after a colon is ordinary; the capital
#                      guard refused to split, the residue stayed fused, and the
#                      real pipeline deleted the whole string)
#      false KEEP   -- "...focused on the incident archive: Production records"
#                      ("records" satisfies a finite-verb approximation while
#                      being a plural NOUN, so a reported topic split into an
#                      asserted-looking fragment and rescued pure bookkeeping)
#    "Capitalised token plus a word ending in s/ed" is not a subject-plus-finite
#    -verb discriminator, and no lexical test at this layer is. A guard wrong in
#    both directions is not a threshold miss, so it is DELETED rather than
#    retuned. Colons now inform the residue check below instead of splitting.
#
# 2. NO CAPITAL GUARD ON SENTENCE PUNCTUATION, and closing quotes are absorbed.
#    The earlier `(?=[A-Z])` was ASCII-only, so a Greek or Cyrillic sentence
#    start fused back into the occurrence, and a closing quote before the
#    capital did the same. Over-splitting produces more unclassifiable clauses,
#    and unknown keeps -- the safe direction. Version numbers and decimals carry
#    no space after the period, so they still never match.
#
# 3. CASE-INSENSITIVITY IS LOCAL, NOT PATTERN-WIDE. An earlier revision dropped
#    a pattern-level `(?i)` to make `[A-Z]` guards work and silently made the
#    PRE-EXISTING word branches case-sensitive too, so an uppercase "AND" or a
#    capitalised "Which" stopped splitting and deleted durable residue behind
#    them. Inline `(?i:...)` restores those branches without weakening anything
#    that needs case.


# A proposition needs a predicate. Coordination joins NOUNS at least as often as
# it joins clauses ("the first and last recorded events"), and splitting there
# produces a verbless fragment that is not a proposition at all -- it would be
# unclassifiable, and unknown keeps, so a journal row would survive on a
# grammatical accident. Fragments with no verb are re-joined to their neighbour.
_VERBISH_RE = re.compile(
    r"(?i)\b(?:is|are|was|were|be|been|being|has|have|had|do|does|did"
    r"|will|would|can|could|may|might|must|should|shall|ought"
    r"|\w+ed|\w+ing|\w+es|\w+s)\b"
)


_ASCII_ONLY_RE = re.compile(r"^[\x00-\x7f]*$")


def _looks_verbless(fragment: str) -> bool:
    """True if *fragment* has no predicate AND this heuristic can actually read it.

    THE ASCII CONDITION IS THE POINT, and it was found by measurement rather than
    by reading the code. A reviewer's Greek probe kept fusing back into its
    occurrence clause, and the splitter was innocent -- it split correctly and
    the FOLD undid it. ``_VERBISH_RE`` recognises predicates by ASCII suffixes
    (-ed, -ing, -es, -s), so any non-Latin clause looks verbless, gets folded
    backwards into the occurrence, and its durable residue is deleted with it.
    The ASCII assumption sat one layer BELOW the ``[A-Z]`` guards everyone was
    looking at.

    So a fragment this heuristic cannot read is not judged by it: no fold, the
    fragment stands as its own clause, and an unclassifiable clause keeps.
    """
    return bool(_ASCII_ONLY_RE.match(fragment)) and not _VERBISH_RE.search(fragment)


def _split_into_propositions(content: str) -> list[str]:
    """Approximate the propositions in *content*. See the bias note above."""
    parts = [c.strip() for c in _PROPOSITION_SPLIT_RE.split(content) if c and c.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and _looks_verbless(part):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    # A leading verbless fragment has no earlier neighbour to join; fold it forward.
    while len(merged) > 1 and _looks_verbless(merged[0]):
        merged[1] = f"{merged[0]} {merged[1]}"
        merged.pop(0)
    return merged


# Attitude / reporting predicates. Their COMPLEMENT is class C: mentioned, not
# asserted. This list is a partial recognizer for a semantic category, not the
# definition of it -- what it fails to recognize becomes unknown, and unknown
# keeps, so the list being incomplete is safe in the direction that matters.
_REPORTING_PREDICATE_RE = re.compile(
    r"(?i)\b(?:"
    r"focus(?:ed|sed|es|ing)?\s+on|focus\s*:|focus\s+for"
    r"|w(?:as|ere)\s+about|discussed|covered|centred\s+on|centered\s+on"
    r"|regarding|concerning|on\s+the\s+topic\s+of|with\s+focus\s+on"
    r")"
)


def _asserted_part(clause: str) -> str:
    """The clause with its reported complement removed.

    THIS IS THE FIX FOR THE KEEP-SIDE LEAK. Scanning a whole clause for
    standing-rule vocabulary counts words that sit inside what a session merely
    focused ON. "focused on what should happen next" is not an assertion that
    anything should happen; the modal belongs to the reported topic. Marker
    scans run on THIS string, never the raw clause.
    """
    m = _REPORTING_PREDICATE_RE.search(clause)
    return clause[: m.start()] if m else clause


def _is_reported_complement_only(clause: str) -> bool:
    """CLASS C: the clause is a reporting label plus its complement, with no
    occurrence anchored to it.

    "Focus: visible focus rings are required for keyboard accessibility" is a
    topic label on a durable accessibility rule -- inert, not condemning, which
    is why C alone can never reject. Contrast "Session a1b2c3d4 focused on the
    policy review", where the reporting predicate IS anchored to a session id:
    that clause is class B, and its complement is stripped rather than scanned.
    """
    if _SESSION_ID_RE.search(clause):
        return False
    m = _REPORTING_PREDICATE_RE.search(clause)
    return bool(m and not _asserted_part(clause).strip(" :,-"))


# Boundary-shaped material that should NOT still be sitting inside a clause the
# segmenter is about to call pure occurrence metadata.
_BOUNDARY_RESIDUE_RE = re.compile(r":\s+\S|[.!?][\"'”’\)\]]*\s+\S")


def _has_unsplit_boundary_residue(clause: str) -> bool:
    """True if *clause* still contains material that looks like a boundary.

    THE POINT IS TO KNOW WHEN WE MIGHT BE WRONG, not to split everything. The
    set of punctuation forms that can separate two propositions has no bound a
    lexical layer can enumerate -- reviewers produced closing quotes, a U+2015
    horizontal bar, a closed em dash, non-Latin capitals, and lowercase prose
    after a colon, and there is no reason to believe that list is finished.
    Chasing forms one at a time is a race with no finish line, and every form
    missed deletes a durable fact silently.

    So the segmenter splits only what it is confident about, and this asks the
    other question: does anything boundary-SHAPED remain inside a clause that is
    about to be judged pure occurrence metadata? If so the clause may be two
    propositions wearing one coat, its class-B reading is unsafe, and it is
    demoted to unknown -- which keeps. Segmentation failure becomes FAIL-OPEN in
    the direction the contract already chose.

    STRICTLY ONE-DIRECTIONAL. This runs DOWNSTREAM of segmentation, on residual
    clauses, never as a pre-scan over the raw string, and it can only turn a
    reject into a keep. A pre-scan would be a different function with different
    failure modes: it could see a boundary the segmenter WOULD have split and
    demote content that was never at risk.

    The cost is real and measured: a multi-sentence pure bookkeeping string with
    an internal colon now survives. See limitation (d) in ``_is_metadata_noise``.
    """
    return bool(_BOUNDARY_RESIDUE_RE.search(_asserted_part(clause)))


def _is_occurrence_metadata(clause: str) -> bool:
    """CLASS B: positively identified occurrence metadata -- time, duration,
    topic, command execution -- with no durable result.

    Positive identification is required, not inferred from the absence of
    anything else. A clause this function cannot name is unknown, and the
    caller keeps it.
    """
    # A standing-rule marker in ASSERTED position makes this class A. Reported
    # position does not count -- that is the whole point of _asserted_part.
    if _STANDING_RULE_RE.search(_asserted_part(clause)):
        return False
    return (
        _is_session_record(clause)
        or _is_command_invocation_log(clause)
        or _is_ephemeral_execution_event(clause)
    )


def _is_metadata_noise(content: str) -> bool:
    """Return True if *content* RECORDS AN OCCURRENCE -- one session, one command
    invocation, one deletion -- rather than stating something that still holds.

    See the axis note above ``_STANDING_RULE_RE`` for why this is not a keyword
    or token filter, and why the burden of proof sits on the reject side.

    HONEST LIMITATIONS, in the order they will bite:

    1. Grammar is approximated LEXICALLY, with no parser. Three shapes of false
       keep follow from that, and all three are the chosen direction:

       (a) Quantification INSIDE one occasion reads like quantification over
           occasions. "Every entry logged during session 5517903 pertained to
           the same scratch directory" quantifies over one session's log lines.
       (b) Embedded rule vocabulary in an EPISODIC clause. "Session 9ab4f012
           concluded at 17:42; any remaining follow-up items should be picked
           up" asserts the modal in its own right, so it is class A by the
           contract even though the content is a journal row. Note this is
           NOT the reported-position case: "focused on what should happen next"
           mentions the modal inside a complement and is correctly rejected.
       (d) BOOKKEEPING WHOSE COLON-BEARING CLAUSE IS NOT AN ANCHORED WHOLE-
           STRING TUPLE. "Focus: Run this exact shell command and report its
           exit code: rm -rf /tmp/scratch-run-00000" is genuine junk and
           survives, because a colon with material after it inside a class-B
           clause is the residue signal, and demotion keeps.

           THE QUALIFIER IS LOAD-BEARING and this limitation was briefly
           written without it. "Multi-sentence bookkeeping with an internal
           colon" would also describe "Session deadbeef: 2026-08-01", which
           the anchored-tuple bypass REJECTS. A limitation section describes
           what the code cannot do, so it goes stale in the direction of
           UNDERSTATING the code -- harmless for a user, misleading for a
           reviewer, who reads it as the current failure set.

           This is the PRICE OF DELETING the conditional colon split, paid
           deliberately and ratified in review. That guard was proven wrong in
           BOTH directions: it refused to split lowercase prose after a colon
           (deleting a durable production invariant) and it DID split a plural
           noun read as a finite verb (rescuing bookkeeping). Keeping it to
           preserve this one rejection would have retained a discriminator that
           demonstrably deletes durable clauses -- the wrong side of the
           asymmetry, one case smaller.

       (c) A PARTICIPIAL ADJUNCT that is shaped like a residue but carries
           none. "...was removed at the close of the run, leaving no residual
           files behind" separates into a clause this layer cannot classify,
           and unknown keeps.

       (c) is unavoidable given (and worth) the amendment that makes result-
       bearing occurrences survive: "/tmp/build-cache was removed after upload
       failed, exposing the cleanup race that truncated release manifests" has
       the IDENTICAL participial shape and must keep. Letting a participial
       inherit its host clause's class would reject both. No lexical test
       separates them without re-introducing the token discriminator this
       design exists to remove, so the cost is paid deliberately.

       MEASURED on an adversarial corpus built to exploit exactly this: of 14
       bookkeeping items inside the three shapes, 6 are rejected and 8 survive
       -- 6 by (a)/(b) and 2 by (c). The earlier whole-string version rejected
       8, and it bought those two extra rejections by deleting every
       result-bearing occurrence, which is the defect this revision closes.
       Demoting the keep evidence was also tried and measured: it bought 4 more
       rejections at the price of 5 durable facts, including "time to first
       byte is measured from the edge log line where session 9c4e1f occurred,
       never from the client clock". Both trades run the wrong way.
    2. A model that rewords a transient event into rule-shaped declarative prose
       slips past any fixed pattern set. So a low rejection count is NOT evidence
       of clean output -- verify quality by READING nodes, never by this filter's
       own counter, which is circular.
    3. The three shapes are the ones this layer has observed. Content that is
       bookkeeping in some fourth shape is not covered, and adding a shape
       requires a pinned reject case first.

    Note on where the primary defence belongs: suppressing this class at the
    PROMPT is strictly better than filtering it after the fact, but recall no
    longer owns the Stage-1 prompt -- ``extract_batch`` builds it from the
    extract package's own fragments, and recall passes only units, infer,
    produced_by and capabilities. So until that forbid-list exists upstream,
    this filter is the sole defence at this layer rather than a second one.
    """
    saw_occurrence_metadata = False
    for clause in _split_into_propositions(content):
        if _is_occurrence_metadata(clause):
            # BYPASS: an EXACT whole-string journal tuple is not ambiguous, so
            # there is nothing for the detector to be uncertain about.
            #
            # Residue demotion reopened the round-1 core case:
            # "Session deadbeef, 2026-08-01" rejected while
            # "Session deadbeef: 2026-08-01" kept, same clause, because a colon
            # fired the detector. A safety net that opens the case it was built
            # over is worse than the gap it covered.
            #
            # The fix is NOT a lexical test for what follows the colon. That
            # approximation was already proven wrong in both directions as a
            # split discriminator, and moving it into demotion would preserve
            # its destructive failure direction while changing where it lives.
            # There is deliberately NO short-token exemption either: length is
            # not evidence that something is metadata.
            #
            # What licenses the bypass is that _SESSION_RECORD_TUPLE_RE is
            # ANCHORED at both ends. A whole-string match means the clause is
            # ENTIRELY an id and a timestamp, with no unaccounted material for a
            # missed boundary to be hiding in. Every ambiguous colon still keeps.
            if not _SESSION_RECORD_TUPLE_RE.match(clause) and \
                    _has_unsplit_boundary_residue(clause):
                # The segmenter may have missed a split inside this clause, so
                # its class-B reading is unsafe. Demote to UNKNOWN, and unknown
                # keeps. Never the reverse: residue can only turn a REJECT into
                # a keep, never a keep into a reject.
                return False
            saw_occurrence_metadata = True
        elif _is_reported_complement_only(clause):
            continue  # class C is inert: it neither rescues nor condemns
        else:
            # Class A, or a clause this layer cannot classify. Both keep.
            return False
    # C-only content reaches here with the flag still False and is KEPT: a
    # reported topic on its own is not evidence that anything is bookkeeping.
    return saw_occurrence_metadata


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


def _normalize_create_content(raw_content: str) -> str:
    """The exact normalization ``_apply_consolidation_result``'s create branch runs before
    any filter sees the content: truncate to 300 chars, scrub, strip markdown formatting,
    strip section-header prefixes. Factored out so B4 (``_evaluate_create_content`` below)
    can run the IDENTICAL sequence on its own composed candidates — a raw string that isn't
    whitespace can still normalize down to empty or low-specificity (e.g. a pure markdown/
    section-prefix wrapper), which only checking the raw string misses (Sentinel, recall#884
    re-review, fruit 2: a 152-char "Operational Deployment Readiness: ..." composition strips
    to a 118-char low-specificity remainder at real B3)."""
    content = scrub_text(_tw(str(raw_content), 300))
    content = strip_markdown_formatting(content)
    content = _strip_section_prefix(content)
    return content


def _create_content_passes_filters(
    content: str, content_profile=None, *, is_create: bool = True,
) -> bool:
    """The rejection filters ``_apply_consolidation_result``'s create branch runs, extracted
    into the SAME function both B3's create branch and B4 call — parity by construction,
    immune to drift (Sentinel, recall#884 re-review: B4's own hard-coded
    ``threshold=120``/``content_type=None`` diverged from what B3 actually evaluates once
    ``content_profile`` flows in — which it already does at B3, via ``_run_extract_path``,
    just not at B4).

    ``is_create`` matches the original inline code's own ``action == "create"`` gating: the
    generic/specificity/garbled checks only ever applied to create-action nodes in B3, but
    the few-shot contamination check ("[PersonA]"/"[PersonB]") is UNCONDITIONAL there — it
    also protects corroborate/contradict content. B4 only ever evaluates create candidates,
    so it always passes ``is_create=True`` (the default); B3 passes the real
    ``action == "create"`` so corroborate/contradict keep their exact prior behavior
    (contamination-checked, generic/specificity/garbled-exempt).

    ``_is_metadata_noise`` is the second UNCONDITIONAL check, for the reason spelled out at
    its call site below: ``is_create`` is the model's action LABEL, not a statement about
    whether this path lets the candidate become, replace, or corroborate knowledge.

    Assumes *content* is ALREADY normalized (``_normalize_create_content``) and non-empty —
    callers run that + the empty check themselves (B3 already does, unconditionally, for
    every action, not just create; see ``_evaluate_create_content`` for B4's full pipeline).
    """
    if is_create:
        _ap = None
        if content_profile is not None:
            from synapt.recall.content_profile import adaptive_params
            _ap = adaptive_params(content_profile)

        if _ap is None or _ap.generic_filter_enabled:
            if _is_generic_node(content):
                logger.info("Rejected generic node (pattern): %s", content[:80])
                return False
        spec_threshold = _ap.specificity_threshold if _ap else 120
        _ct = content_profile.content_type if content_profile is not None else None
        if _lacks_specificity(content, threshold=spec_threshold, content_type=_ct):
            logger.info("Rejected generic node (low specificity): %s", content[:80])
            return False

    if "[PersonA]" in content or "[PersonB]" in content:
        logger.info("Rejected example-contaminated node: %s", content[:80])
        return False

    if is_create and _is_garbled_content(content):
        logger.info("Rejected garbled node: %s", content[:80])
        return False

    # UNCONDITIONAL, unlike its three neighbours above, and deliberately so.
    #
    # ``is_create`` names the action the MODEL chose for this candidate. It does
    # NOT name whether this code path lets the candidate's text become knowledge,
    # and those are independent: the create fallthroughs below reach
    # ``action = "create"`` only AFTER this function has already run with
    # ``is_create=False``, so a missing corroborate/contradict target persists the
    # candidate verbatim through a gate that declined to look at it. Legacy
    # (no-DB) contradiction persists it as the new ACTIVE node; DB contradiction
    # queues it as ``new_content``; corroborate-with-target lets it raise a real
    # node's confidence as if it were evidence.
    #
    # Gating a CONTENT-QUALITY property on a ROUTING label is the category error,
    # so the fix is to drop the condition rather than repeat the check on each
    # branch -- one gate that cannot be routed around beats four that can.
    #
    # The three neighbours stay create-only on purpose: specificity and
    # genericness are judged against creation, and contradictions legitimately
    # reference existing project-specific nodes (see the explicit
    # ``_is_generic_node`` re-check in the contradict branch). Bookkeeping is
    # different in kind -- content that must never BECOME knowledge must also
    # never REPLACE it or COUNT AS EVIDENCE for it, whichever action the model
    # picked. Consistency with the neighbours is not a reason here.
    if _is_metadata_noise(content):
        logger.info("Rejected metadata-noise node (action-independent): %s", content[:80])
        return False
    return True


def _evaluate_create_content(raw_content: str, content_profile=None) -> str | None:
    """The single source of truth for whether ``_apply_consolidation_result``'s create
    branch will accept *raw_content*, and what NORMALIZED string it will persist if so.
    Normalizes (``_normalize_create_content``), rejects if empty, then runs the same 4
    filters B3 runs (``_create_content_passes_filters``). Returns the normalized content on
    acceptance, ``None`` on rejection.

    This is B4's own entry point (it has no already-normalized ``content`` variable lying
    around the way B3's loop body does) — B4 uses ONLY the verdict (whether this returns
    ``None``), never the normalized string itself, so B3 remains the single writer/
    normalizer of persisted content at apply time.
    """
    content = _normalize_create_content(raw_content)
    if not content:
        return None
    if not _create_content_passes_filters(content, content_profile):
        return None
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

# Fix B contested-memory-lifecycle reframe (internal design spec, section 10.5/10.10
# item 5): confidence ceiling applied to both
# nodes in a contested pair. Belt-and-suspenders, not the primary mechanism -- the primary
# exclusion is the existing status != 'active' FTS gate (storage.py's knowledge_fts_search),
# which hides contested nodes from default search entirely. This ceiling only matters when a
# contested node is explicitly surfaced via include_historical=True: below the existing 0.4
# no-boost gate (core.py's _search_knowledge), so a contested node never gets a ranking boost
# regardless of its original confidence. Named constant so the dogfood run can tune it.
_CONTESTED_CONFIDENCE_CEILING = 0.3

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


# A7 fix (internal design spec; Opus's 2026-07-17 ruling on the PR #892 1c
# finding): containment for the create-branch dedup
# decision is a contiguous TOKEN-subsequence match, not a raw character substring match.
# Punctuation is stripped ONLY at token boundaries (leading/trailing); punctuation INSIDE a
# token is preserved, so "v1.3" and "v1.35" stay distinct atomic tokens (never collapse into
# each other via a shared numeric prefix) while "hold;" and "hold," reduce to the same token
# "hold" (a clause-boundary separator carries no content of its own). "?" is deliberately
# EXCLUDED from the strip set: a trailing question mark changes the speech act (an assertion
# and a question about the same words are not the same claim), so "?" is preserved as part of
# its token rather than stripped -- see test_a7_question_form_never_merges_with_assertion.
#
# "!" IS included (Opus's ruling on Sentinel's reviewer-2 record/code mismatch, 2026-07-17):
# unlike "?", an exclamation mark does not change what KIND of utterance a sentence is (still
# an assertion, not a question) -- it marks emphasis/register on an otherwise identical claim,
# boundary noise the same way ";" is. "the deploy failed" and "the deploy failed!" are the
# same claim at different volume -- see test_a7_exclamation_is_boundary_noise_not_a_new_claim.
_TOKEN_BOUNDARY_STRIP = ".,;:!"


def _tokenize_for_containment(text: str) -> list[str]:
    """Lowercase + whitespace-split, then strip only `_TOKEN_BOUNDARY_STRIP` characters from
    each token's own leading/trailing edge. Intra-token characters (decimals, hyphens,
    underscores, slashes, question marks) are never touched, so a real prefix like "--verbose"
    or "-x" is preserved whole -- `_TOKEN_BOUNDARY_STRIP` deliberately excludes "-" for exactly
    this reason (Sentinel's reviewer-2 caution, 2026-07-17: never strip a meaningful CLI-flag
    prefix while fixing standalone separators).
    A punctuation-only "word" strips to an empty string via the normal boundary strip (a spaced
    ellipsis: "landed . . . eventually" -> the isolated "." tokens vanish) and is dropped rather
    than kept as a positional placeholder that would pollute a contiguous-subsequence match with
    a stray gap the other text has no counterpart for. A STANDALONE dash/em-dash token ("a - b")
    is caught separately: it never touches `_TOKEN_BOUNDARY_STRIP` (would strip "--verbose"),
    so a token that strips to something made ENTIRELY of "-" characters is treated as empty too
    -- see test_a7_standalone_dash_separator_is_not_content."""
    tokens: list[str] = []
    for raw in text.lower().split():
        tok = raw.strip(_TOKEN_BOUNDARY_STRIP)
        if not tok or set(tok) <= {"-"}:
            continue
        tokens.append(tok)
    return tokens


def _token_sequence_contains(shorter: list[str], longer: list[str]) -> bool:
    """True if `shorter` appears as a CONTIGUOUS run inside `longer`, token-for-token (not a
    character substring). An empty `shorter` is trivially contained, mirroring `"" in s`."""
    if not shorter:
        return True
    n = len(shorter)
    return any(longer[i:i + n] == shorter for i in range(len(longer) - n + 1))


def _b3_containment_decision(candidate_content: str, existing_content: str) -> str:
    """The create-branch auto-corroborate decision rule, replacing the old unconditional
    threshold-crossing convert-to-corroborate. Returns "keep_existing" (candidate's claim is
    already fully covered by existing -- corroborate, never touching content), "supersede"
    (existing's claim is fully covered by candidate, which says strictly more -- mark existing
    superseded, persist the fuller content as a new node), or "keep_both" (neither contains the
    other -- skip the auto-corroborate branch entirely, fall through to the ordinary create
    path unchanged)."""
    cand_tokens = _tokenize_for_containment(candidate_content)
    exist_tokens = _tokenize_for_containment(existing_content)
    if _token_sequence_contains(cand_tokens, exist_tokens):
        return "keep_existing"
    if _token_sequence_contains(exist_tokens, cand_tokens):
        return "supersede"
    return "keep_both"


def _representative_source_unit_id(raw_node: dict) -> "str | None":
    """Fix B (internal design spec, section 3): reads a B4 action-item dict's chronology
    signal. A singleton pass-through carries
    ``source_unit_id`` (singular, threaded end-to-end from ``_decide_actions``). A composed
    group carries ``source_unit_ids`` (plural, one per member — see the ``"source_unit_ids":
    [m.get("source_unit_id") for m in members]`` shape at compose time). Returns the LATEST
    member's value for a composed group (Opus's r1 decision, spec section 9.2): a composed
    node represents the current synthesized understanding, so biasing its effective
    chronology toward its most recent contributing fact makes it HARDER to supersede later —
    fewer false-supersedes, the failure mode most worth avoiding under constraint 2's
    asymmetric conservatism. ``None`` for any shape without a usable value (legacy/collection
    paths never set either key) — never fabricated, never a proxy."""
    singular = raw_node.get("source_unit_id")
    if singular is not None:
        return singular
    plural = raw_node.get("source_unit_ids")
    if not plural:
        return None
    candidates = [s for s in plural if s is not None]
    if not candidates:
        return None
    latest = candidates[0]
    for candidate in candidates[1:]:
        if _source_unit_id_order(candidate, latest) == 1:
            latest = candidate
    return latest


def _b3_temporal_conflict_escalation(
    *,
    candidate_content: str,
    candidate_source_unit_id: "str | None",
    existing_content: str,
    existing_source_unit_id: "str | None",
    conflict_judge,
) -> str:
    """Fix B (internal design spec,
    section 10 -- Layne-ratified contested-memory-lifecycle reframe, 2026-07-21): only called
    when ``_b3_containment_decision`` already returned ``"keep_both"`` — neither text contains
    the other. Escalates to ``"contest"`` ONLY when the judge reports a genuine conflict; any
    missing signal or judge uncertainty falls through to ``"keep_both"`` unchanged.
    Asymmetric-conservative by construction: every early-return here is ``"keep_both"``, never
    a guessed winner.

    The judge is a FLAGGER, not a resolver (section 10.1) — ``"contest"`` is never auto-applied
    by the caller; it always routes to the contested-memory lifecycle (mark both nodes
    contested, queue for human/agent review via ``add_pending_contradiction``). This is why
    the contract no longer distinguishes ``"supersede"`` from ``"keep_existing"``: chronology
    direction is no longer used to pick an automatic winner here, only to annotate the queued
    review with reviewer context (built by the caller via ``_source_unit_id_order``, not by
    this function). Section 10.9/10.10 item 1: the same-entry-tie guard is DROPPED for this
    reason — under contest, a tie needs no resolvable direction to be worth flagging; the
    reviewer decides on content. Item 2: the missing-``source_unit_id`` guard STAYS — without
    any chronology signal at all, a reviewer gets no useful context, and a node from a path
    that doesn't thread the field (legacy, collection pass) is a documented, deliberate
    known-limitation exclusion, not an oversight.

    ``conflict_judge: Callable[[str, str], bool | None] | None`` — injected, not hardwired
    (mirrors the ``infer`` seam B1/B2/B4 already use). ``None`` return from the judge means
    genuinely uncertain and is treated identically to ``False``.
    """
    if conflict_judge is None:
        return "keep_both"
    if candidate_source_unit_id is None or existing_source_unit_id is None:
        return "keep_both"
    is_conflict = conflict_judge(candidate_content, existing_content)
    if not is_conflict:  # False, or None (uncertain) -- both mean "not proven"
        return "keep_both"
    return "contest"


def _local_conflict_judge(infer):
    """Production default for ``_b3_temporal_conflict_escalation``'s ``conflict_judge``
    parameter. Wraps the extract path's own already-injected ``infer`` seam
    (``_make_recall_infer``) — local-first by construction, since ``infer`` IS the local
    MLX/enrichment model this whole pipeline already runs on, never a new cloud dependency.
    Only ever invoked from inside the caller's already-narrow high-similarity-non-containment
    trigger band (constraint 1) — not on every ``keep_both``.

    KNOWN, DOCUMENTED, LOW-STAKES GAP (internal design spec, section 9.4, reframed
    section 10.10 item 4): this exact wording — the
    best of 4 tried, across both 4-bit and bf16 Ministral-3-3B — reliably classifies the
    founding case (fixture a, REQUIRED) and a negation-flip (fixture d) correctly, but is NOT
    reliable on same-entity-different-property pairs sharing strong lexical overlap without a
    genuine logical conflict (fixture b) — it over-weights topical/lexical overlap as evidence
    of conflict even when its own stated reasoning correctly identifies the two claims as
    distinct properties. This is a real, reproducible model-capacity limit at this scale, not
    a parsing or prompt-triviality issue; see the spec addendum for the full investigation.
    Under the contested-memory-lifecycle reframe this is a review-queue-noise issue, not a
    safety issue (tracked: recall#900) — a false CONFLICT call here costs a confidence dip and
    a queue entry on both nodes, never a lost fact; the judge is a flagger, not a resolver.

    Parsed strictly and conservatively: anything other than a response that starts with
    exactly "CONFLICT" (case-insensitive, after stripping whitespace) returns ``None``
    (uncertain), never ``True`` — a malformed response, an unexpected model utterance, or an
    inference exception must never be silently treated as evidence of conflict.
    """
    def judge(candidate_content: str, existing_content: str) -> "bool | None":
        prompt = (
            "Two statements, possibly about the same general topic. Apply this exact "
            "test: if Statement A is true, does that make Statement B FALSE (or vice "
            "versa)? Sharing a topic is not enough -- they must actually contradict.\n\n"
            f"Statement A: {candidate_content}\n"
            f"Statement B: {existing_content}\n\n"
            "If A being true would make B false (or B true would make A false): answer "
            "CONFLICT.\n"
            "If both statements can be true at the same time, even though they describe "
            "different specific details of the same topic: answer COMPATIBLE.\n\n"
            "Answer with exactly one word: CONFLICT or COMPATIBLE."
        )
        try:
            response = infer({"messages": [{"role": "user", "content": prompt}]})
        except Exception:
            return None
        answer = (response or "").strip().upper()
        if answer.startswith("CONFLICT"):
            return True
        if answer.startswith("COMPATIBLE"):
            return False
        return None  # unparseable/malformed -- uncertain, not a guess
    return judge


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
    reason: str = "",
) -> None:
    """Append one pairwise decision to the dedup decisions JSONL file.

    Pure logging — never disrupts consolidation.

    *reason* (Fix B, internal design spec):
    for ``auto-corroborate``/``auto-supersede``/``contest`` entries, distinguishes WHICH
    mechanism produced the decision — ``"containment"`` (A7's token-subsequence rule) vs
    ``"chronology_contest"`` (section 10's contested-memory-lifecycle escalation: judge
    flags CONFLICT, no direction is auto-applied) — matching guard 6's own precedent of
    logging *why*, not just *what*.
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
    if reason:
        entry["reason"] = reason

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


def _corroborate_bound_fill(target: KnowledgeNode, raw_node: dict) -> dict:
    """BLOCKER 2 fix (Sentinel, 2026-07-15): the FILL-MISSING-bounds-only update-dict fragment
    shared by every corroborate-shaped path — the explicit ``action == "corroborate"`` branch AND
    the create branch's own similarity-triggered auto-corroborate (a THIRD path found in
    self-review, structurally identical: a create-action candidate silently becomes a corroborate
    via Jaccard/cosine match, never touching ``raw_node["action"]``, so it needs the exact same
    fix or the defect just reappears under a different name).

    Each bound filled INDEPENDENTLY, only when *target*'s persisted value is missing (``None``)
    — a conflicting persisted bound is NEVER overwritten. Returns only the keys that should
    change, so the caller's ``update_node`` (a targeted ``dict.update()``) leaves everything else
    — including a conflicting bound — untouched.
    """
    updates: dict = {}
    if _env_flag("SYNAPT_DISABLE_TEMPORAL_EXTRACTION"):
        return updates
    if target.valid_from is None:
        candidate_valid_from = _validate_iso_date(raw_node.get("valid_from"))
        if candidate_valid_from:
            updates["valid_from"] = candidate_valid_from
    if target.valid_until is None:
        candidate_valid_until = _validate_iso_date(raw_node.get("valid_until"))
        if candidate_valid_until:
            updates["valid_until"] = candidate_valid_until
    return updates


def _apply_corroborate_update(target: KnowledgeNode, updates: dict, knowledge_path: Path) -> bool:
    """Apply *updates* to the persisted node via ``update_node``, and — ONLY on success —
    mutate *target* in place to match (Sentinel's re-clear finding, 2026-07-15).

    ``existing_by_id``/``existing_nodes`` are built ONCE at the top of
    ``_apply_consolidation_result`` and never re-read from disk mid-batch. Without this sync, TWO
    candidates in the SAME batch that both corroborate the SAME node each compute their bound-fill
    from the SAME stale ``target`` object: candidate 1's ``update_node`` call appends a fresh
    persisted version, but candidate 2 still sees the pre-batch ``target.valid_until is None`` and
    silently overwrites candidate 1's now-persisted fill — reproduced through the real B1->B2->B3
    path (two prefilter candidates corroborating one node, first candidate's bound lost). Mutating
    ``target`` here — the SAME object both ``existing_by_id`` and ``existing_nodes`` reference —
    means a later same-batch candidate targeting this node sees the just-applied state, restoring
    fill-missing/never-overwrite-conflicting WITHIN a batch, not just across separate calls.

    "Never mutate memory if persistence failed": if ``update_node`` returns ``False`` (target not
    found on disk), *target* is left untouched — an in-memory node claiming a bound that was never
    actually persisted would be worse than the staleness this fixes.
    """
    ok = update_node(target.id, updates, knowledge_path)
    if ok:
        for key, value in updates.items():
            setattr(target, key, value)
    return ok


def _apply_consolidation_result(
    parsed: dict,
    existing_nodes: list[KnowledgeNode],
    cluster: list[JournalEntry],
    knowledge_path: Path,
    decision_log_path: Path | None = None,
    db=None,
    content_profile=None,
    conflict_judge=None,
) -> ConsolidationResult:
    """Apply parsed LLM output: create, corroborate, or contradict nodes.

    When *db* (RecallDB) is provided, contradictions are queued as
    pending_contradictions for user review instead of auto-applied.

    *conflict_judge* (Fix B, internal design spec): optional, defaults to None. Only
    the extract path
    (``_run_extract_path``) passes this — the legacy and collection-pass callers never do,
    so ``_b3_temporal_conflict_escalation`` is always a no-op for them (falls through to
    ``keep_both``, byte-identical to pre-Fix-B behavior). See ``_local_conflict_judge`` for
    the production default.
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
        # Normalize + filter via the shared gate B4 also calls (_evaluate_create_content) —
        # parity by construction, see that function's docstring. is_create mirrors the
        # original per-check action=="create" gating exactly (contamination stays
        # unconditional; generic/specificity/garbled stay create-only).
        content = _normalize_create_content(str(raw_node.get("content", "")))
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

        # Fix B (section 3): the candidate's true source chronology, singleton or composed —
        # None on any path that doesn't thread it (legacy, collection pass).
        candidate_source_unit_id = _representative_source_unit_id(raw_node)

        if not content:
            continue

        if not _create_content_passes_filters(
            content, content_profile, is_create=(action == "create"),
        ):
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
                updates: dict = {
                    "source_sessions": new_sources,
                    "confidence": new_confidence,
                }
                updates.update(_corroborate_bound_fill(target, raw_node))
                _apply_corroborate_update(target, updates, knowledge_path)
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
            # BLOCKER 2 fix (Sentinel, 2026-07-15): contradict CARRIES candidate bounds onto the
            # replacement node + the queued-contradiction payload — validated once, shared by
            # both sub-paths below. Disabled uniformly by the same flag create/corroborate use.
            if _env_flag("SYNAPT_DISABLE_TEMPORAL_EXTRACTION"):
                candidate_valid_from = None
                candidate_valid_until = None
            else:
                candidate_valid_from = _validate_iso_date(raw_node.get("valid_from"))
                candidate_valid_until = _validate_iso_date(raw_node.get("valid_until"))
            if target and db is not None:
                # Queue for user review instead of auto-applying. Bounds carried into the
                # payload so a confirm-time materialization (server.py's _apply_supersession)
                # can apply them — they would otherwise be lost the moment this is queued.
                db.add_pending_contradiction(
                    old_node_id=target.id,
                    new_content=content,
                    category=category,
                    reason=contradiction_note,
                    source_sessions=cluster_sessions,
                    detected_by="consolidation",
                    valid_from=candidate_valid_from,
                    valid_until=candidate_valid_until,
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
                # Candidate's bound preferred (mirrors the create branch's exact logic); falls
                # back to cluster_valid_from/now only when the candidate supplies nothing.
                new_node.valid_from = candidate_valid_from or cluster_valid_from or now
                if candidate_valid_until:
                    new_node.valid_until = candidate_valid_until
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
            # Moved up from just before the ordinary-create call below (Fix B contest branch,
            # section 10.5, needs it too -- computed once, used by both).
            confidence = raw_node.get("confidence", 0.5)
            if not isinstance(confidence, (int, float)):
                confidence = 0.5

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
                # A7 fix (internal design spec; Opus's 2026-07-17 ruling): a similarity
                # match no longer unconditionally
                # converts to corroborate. Content containment decides the outcome — the old
                # unconditional path silently discarded any candidate content that wasn't an
                # exact restatement of `best_match`, a live production bug (17/31
                # auto-corroborates in one A6 dogfood run).
                b3_decision = _b3_containment_decision(content, best_match.content)
                decision_reason = "containment"

                if b3_decision == "keep_both":
                    # Fix B (internal design spec, section 10 -- contested-memory-lifecycle
                    # reframe):
                    # containment found neither side contains the other, but that alone
                    # doesn't mean the two are compatible -- escalate to CONTEST (never an
                    # auto-applied winner) when a genuine conflict is detected. No-op (stays
                    # keep_both) whenever conflict_judge is None or either side lacks a
                    # source_unit_id -- asymmetric-conservative by construction, see the
                    # function's own docstring for every early-return path.
                    escalated = _b3_temporal_conflict_escalation(
                        candidate_content=content,
                        candidate_source_unit_id=candidate_source_unit_id,
                        existing_content=best_match.content,
                        existing_source_unit_id=best_match.source_unit_id,
                        conflict_judge=conflict_judge,
                    )
                    if escalated != "keep_both":
                        b3_decision = escalated
                        decision_reason = "chronology_contest"

                if b3_decision == "keep_existing":
                    logger.info(
                        "Auto-corroborate (%s=%.2f, keep_existing): %s",
                        best_method, best_sim, content[:80],
                    )
                    new_sources = list(set(best_match.source_sessions + cluster_sessions))
                    new_confidence = compute_confidence(len(new_sources))
                    auto_updates: dict = {
                        "source_sessions": new_sources, "confidence": new_confidence,
                    }
                    # BLOCKER 2 fix (Sentinel, 2026-07-15, third path found in self-review): this
                    # create-action candidate just got silently converted to corroborate by
                    # similarity — needs the SAME fill-missing-bound treatment as the explicit
                    # corroborate branch (_corroborate_bound_fill), or the fix doesn't reach here.
                    auto_updates.update(_corroborate_bound_fill(best_match, raw_node))
                    _apply_corroborate_update(best_match, auto_updates, knowledge_path)
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
                            reason=decision_reason,
                        )
                    continue

                if b3_decision == "supersede":
                    logger.info(
                        "Auto-supersede (%s=%.2f): %s",
                        best_method, best_sim, content[:80],
                    )
                    # Candidate strictly extends existing's claim (non-lossy gain) — persist the
                    # fuller content as a NEW node and mark existing superseded. Never a bare
                    # in-place overwrite: same persistence shape guard 6 already uses for
                    # same-batch supersession, so the existing node's own history stays
                    # inspectable rather than silently replaced.
                    new_sources = list(set(best_match.source_sessions + cluster_sessions))
                    new_tags = sorted(set(list(best_match.tags) + tags))
                    superseding_node = KnowledgeNode.create(
                        content=content,
                        category=category,
                        source_sessions=new_sources,
                        confidence=compute_confidence(len(new_sources)),
                        tags=new_tags,
                        source_turns=source_turns,
                        source_unit_id=candidate_source_unit_id,
                    )
                    if _env_flag("SYNAPT_DISABLE_TEMPORAL_EXTRACTION"):
                        llm_valid_from = None
                        llm_valid_until = None
                    else:
                        llm_valid_from = _validate_iso_date(raw_node.get("valid_from"))
                        llm_valid_until = _validate_iso_date(raw_node.get("valid_until"))
                    cluster_valid_from = _cluster_valid_from(cluster)
                    superseding_node.valid_from = (
                        llm_valid_from
                        or cluster_valid_from
                        or datetime.now(timezone.utc).isoformat()
                    )
                    if llm_valid_until:
                        superseding_node.valid_until = llm_valid_until
                    update_node(
                        best_match.id,
                        {"status": "superseded", "superseded_by": superseding_node.id},
                        knowledge_path,
                    )
                    append_node(superseding_node, knowledge_path)
                    existing_nodes.append(superseding_node)  # track for intra-batch dedup
                    result.nodes_created += 1
                    if decision_log_path:
                        _log_dedup_decision(
                            decision_log_path,
                            action="auto-supersede",
                            candidate_content=content,
                            candidate_category=category,
                            existing_id=best_match.id,
                            existing_content=best_match.content,
                            similarity_score=best_sim,
                            source=f"auto-{best_method}",
                            session_ids=cluster_sessions,
                            reason=decision_reason,
                        )
                    continue

                if b3_decision == "contest" and db is not None:
                    # Contested-memory-lifecycle reframe (internal design spec, section
                    # 10.5, Layne-ratified
                    # 2026-07-21): the judge is a FLAGGER, not a resolver. Neither side is
                    # auto-applied as a winner -- both nodes persist, both marked contested
                    # with confidence capped, and a pending_contradiction queues the pair for
                    # human/agent review (_apply_contest_resolution, server.py, is the
                    # resolution path). Requires db: without a review mechanism, "contest" has
                    # no meaningful legacy behavior (unlike the "contradict" action-item's
                    # legacy auto-apply, contest's whole point is deferring to a judge that
                    # is NOT the model) -- falls through to the ordinary create path below,
                    # same as keep_both, when db is None.
                    logger.info(
                        "Contest (%s=%.2f): %s",
                        best_method, best_sim, content[:80],
                    )
                    contested_node = KnowledgeNode.create(
                        content=content,
                        category=category,
                        source_sessions=cluster_sessions,
                        confidence=min(confidence, _CONTESTED_CONFIDENCE_CEILING),
                        tags=tags,
                        source_turns=source_turns,
                        source_unit_id=candidate_source_unit_id,
                    )
                    contested_node.status = "contested"
                    append_node(contested_node, knowledge_path)
                    existing_nodes.append(contested_node)  # track for intra-batch dedup

                    # recall#905 (0.17.0 blocker, Opus 2026-07-22, Part 1 -- root cause):
                    # both nodes must ALSO be upserted into SQLite here, immediately, not
                    # left to some later _sync_knowledge_to_db call. Without this, a fresh
                    # contest's candidate node genuinely doesn't exist in SQLite until a sync
                    # happens to run, and _apply_contest_resolution's db.get_knowledge_node
                    # silently returns None the moment someone tries to resolve it before
                    # that -- the dogfood-found false-success bug (Part 2, server.py, makes
                    # that failure loud regardless; this closes the gap at its source so a
                    # fresh contest is resolvable immediately, matching the real
                    # conversation-memory flow where review can happen right after
                    # detection).
                    db.upsert_knowledge_node(contested_node.to_dict())

                    existing_updates = {
                        "status": "contested",
                        "confidence": min(
                            best_match.confidence, _CONTESTED_CONFIDENCE_CEILING
                        ),
                    }
                    update_node(best_match.id, existing_updates, knowledge_path)
                    existing_dict = best_match.to_dict()
                    existing_dict.update(existing_updates)
                    db.upsert_knowledge_node(existing_dict)

                    # Chronology direction is reviewer CONTEXT here, never a resolution --
                    # the escalation function itself no longer computes or uses it (section
                    # 10.5: direction moved from the pure function to this call site).
                    order = _source_unit_id_order(
                        candidate_source_unit_id, best_match.source_unit_id
                    )
                    if order == 1:
                        contest_reason = (
                            "Chronology: candidate (newer) conflicts with existing (older) "
                            "-- judge flagged CONFLICT, escalation does not auto-resolve."
                        )
                    elif order == -1:
                        contest_reason = (
                            "Chronology: candidate (older) conflicts with existing (newer) "
                            "-- judge flagged CONFLICT, escalation does not auto-resolve."
                        )
                    else:
                        contest_reason = (
                            "Same-entry conflict, no chronology tiebreaker available -- "
                            "judge flagged CONFLICT, escalation does not auto-resolve."
                        )
                    db.add_pending_contradiction(
                        old_node_id=best_match.id,
                        new_content=content,
                        new_node_id=contested_node.id,
                        category=category,
                        reason=contest_reason,
                        source_sessions=cluster_sessions,
                        detected_by="b3-temporal-conflict-escalation",
                    )
                    result.nodes_contested += 1
                    if decision_log_path:
                        _log_dedup_decision(
                            decision_log_path,
                            action="contest",
                            candidate_content=content,
                            candidate_category=category,
                            existing_id=best_match.id,
                            existing_content=best_match.content,
                            similarity_score=best_sim,
                            source=f"auto-{best_method}",
                            session_ids=cluster_sessions,
                            reason=decision_reason,
                        )
                    continue

                # b3_decision == "keep_both" (or "contest" with no db to queue against):
                # neither contains the other and nothing was auto-resolved. Deliberately no
                # `continue` here — falls through to the ordinary create path below, exactly as
                # if no similarity match had been found at all. Zero new node-creation logic.

            new_node = KnowledgeNode.create(
                content=content,
                category=category,
                source_sessions=cluster_sessions,
                confidence=min(1.0, max(0.0, confidence)),
                tags=tags,
                source_turns=source_turns,
                source_unit_id=candidate_source_unit_id,
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

    Per internal design ratification: consolidation is the primary site for
    substrate-reshape scoring. This helper
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
# B1 — decomposed extract path, the front half (behind SYNAPT_USE_EXTRACT)
#
# prefilter (identify.py) -> BatchUnits -> extract_batch -> per-unit envelopes. This half
# produces envelopes + logs per-unit failed-unit markers (never silent-dropped); it does not
# itself decide create/corroborate/contradict or touch knowledge.jsonl — B2 (below) computes
# the action, B3 (_run_extract_path's tail) feeds the SAME reconcile the monolith uses. See the
# "B2 — the action-decision pass" block below for the rest of the pipeline. Together, B1+B2+B3
# replace the monolithic CONSOLIDATION_PROMPT pass when the flag is on.
# ---------------------------------------------------------------------------

# provider URI for the extraction envelope's produced_by. validate_extraction REQUIRES a
# `scheme://identifier` URI — the dotted form ("recall.consolidate") fails EVERY unit as
# schema_invalid (verified against landed extract_batch). Scheme = recall's consolidate STEP
# as the producer, stable across the pluggable model (which model ran is known out-of-band).
_EXTRACT_PRODUCED_BY = "recall://consolidate"
# The capability set recall requests from extract_batch (exact Stage-1 schema keys).
# "temporal_refs" INCLUDED (2026-07-15, extract#31 landed the role field): extraction now emits a
# validity ROLE (effective/expiry/range/superseded/point) + resolved date at the BASE temporal_refs
# capability — no separate "temporal_classes" needed. That base-tier reachability is the exact fix
# extract#31 shipped; the original derivation was silently non-functional because role/resolved_end
# used to sit behind temporal_classes (which recall never requested). recall now maps role ->
# valid_from/valid_until DETERMINISTICALLY in _flatten_envelope_facts (_map_temporal_refs_to_bounds),
# with ZERO LLM re-judgment — direction is read once, at extraction, where the source sentence and
# source date live. See the module's TEMPORAL — ROLE note.
_EXTRACT_CAPABILITIES = ["facts", "decisions", "temporal_refs"]


def _get_consolidation_client(max_tokens: int = MIN_RESPONSE_TOKENS):
    """Get a model client via the router (MLX -> Modal -> Ollama), with an MLX fallback.

    Returns None when no backend is available. Shared by the monolithic path and the
    decomposed extract path so both resolve the model the same way.
    """
    from synapt.recall._model_router import get_client, RecallTask
    client = get_client(RecallTask.CONSOLIDATE, max_tokens=max_tokens)
    if client is not None:
        return client
    if not _MLX_AVAILABLE:
        return None
    from synapt._models.mlx_client import MLXClient, MLXOptions
    return MLXClient(MLXOptions(max_tokens=max_tokens))


def _make_recall_infer(client, model: str):
    """Build the SYNC inference seam that extract_batch injects: a
    ``BatchInferRequest -> completion str`` callable wrapping recall's model client.

    The model is a PLUGGABLE parameter — swapping Ministral / Qwen / Gemma is just a
    different ``model`` routed through the same client; extract stays model-agnostic
    (zero recall coupling). extract_batch calls this SYNCHRONOUSLY (never awaited), so it
    stays a plain function.

    ``max_tokens`` is read PER-REQUEST from an optional ``request["max_tokens"]`` key. One
    ``infer`` closure serves B1's per-unit extract_batch calls, B2's per-cluster
    action-decision call, and B4's per-cluster rejoin call — B2/B4 always set their own
    ``request["max_tokens"]`` explicitly (candidate-count-scaled), so this fallback only
    ever fires for B1.

    B1's ``BatchInferRequest`` (extract#33, synapt-extract) never sets ``max_tokens`` — it
    is out of recall's control (extract_batch owns per-unit request construction). The
    fallback used to be a flat ``MIN_RESPONSE_TOKENS`` (800) floor on the theory that B1's
    per-unit responses were small and fixed-shape; the Phase C dogfood (RESULTS.md
    "Truncation read") falsified that on 3 real unit shapes — Qwen hit 8 extract-unit
    length-finishes against exactly those 3 prompts. Reusing ``_estimate_response_budget``
    (already context-aware: gives most short prompts up to ``CONTEXT_BUDGET`` tokens, only
    drops toward the 800 floor as the prompt itself grows large) closes the gap with no new
    estimator — the same fix class B2 already applied to its own flat-floor bug.
    """
    def recall_infer(request) -> str:
        raw_messages = request.get("messages") or []
        messages = [
            Message(role=m.get("role", "user"), content=m.get("content", ""))
            for m in raw_messages
        ]
        if not messages:  # fall back to the flat prompt if the request carried no messages
            messages = [Message(role="user", content=request.get("prompt", ""))]
        max_tokens = request.get("max_tokens") or _estimate_response_budget(
            request.get("prompt", "")
        )
        return client.chat(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens,
        )
    return recall_infer


def _run_coro_blocking(coro):
    """Drive an async coroutine to completion from a sync caller, across BOTH consolidate
    call-sites — which genuinely differ in whether an event loop is already running:

    - CLI (``cmd_consolidate``): main thread, NO event loop → ``asyncio.run`` (the simple path).
    - MCP (``recall_consolidate``): FastMCP 1.27 runs a sync tool INLINE on the event-loop thread
      (verified: ``FuncMetadata.call_fn_with_arg_validation`` does ``return fn(...)`` directly,
      no ``to_thread``), so a loop IS already running on this thread → ``asyncio.run`` would raise
      ``RuntimeError('asyncio.run() cannot be called from a running event loop')``. We offload the
      coroutine to a fresh thread with its own loop and block on ``join``.

    That join holds the server's event-loop thread for the duration of extraction — acceptable
    because the legacy sync ``consolidate()`` already blocks that same thread inline (consolidation
    has always been a blocking call under FastMCP; this changes nothing about that). Errors from
    the worker re-raise on the calling thread. (Sentinel probed the runtime: loop_running=True,
    same_thread=True — this is the current MCP path, not a hypothetical future one.)
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # CLI path: no running loop

    # MCP path: a loop is already running on THIS thread (FastMCP runs the sync tool inline) —
    # asyncio.run would raise, so run the coroutine in a fresh thread with its own loop.
    box: dict = {}

    def _worker():
        try:
            box["result"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 — surfaced on the calling thread below
            box["error"] = exc

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


def _candidate_source_date(cluster: list[JournalEntry], candidate) -> str | None:
    """The source date to anchor a candidate's relative-date resolution: its OWN journal entry's
    timestamp, date-only (``YYYY-MM-DD``). ``candidate.attr["entry_index"]`` indexes back into
    *cluster* — both come from the same prefilter pass, so the index is in range in the normal
    path. FAIL-SAFE: returns None when the index is missing/out-of-range or the entry has no
    timestamp, so the unit is still built with no anchor rather than crashing or fabricating a
    date (an absent anchor renders no ``Resolve relative dates using:`` line — never the literal
    string "None")."""
    ei = candidate.attr.get("entry_index")
    if not isinstance(ei, int) or not (0 <= ei < len(cluster)):
        return None
    ts = cluster[ei].timestamp
    return ts[:10] if ts else None


async def _extract_cluster_units(
    cluster: list[JournalEntry],
    cluster_id: str,
    infer,
    *,
    produced_by: str = _EXTRACT_PRODUCED_BY,
    capabilities: list[str] | None = None,
):
    """B1 front-half: prefilter -> BatchUnits -> extract_batch -> per-unit envelopes.

    The deterministic prefilter (identify.py) reads structured ``done``/``decisions``; each
    Candidate becomes a ``BatchUnit`` with a cluster-namespaced id (``batch_unit_id``);
    extract_batch shapes + validates each into a per-unit SynaptExtraction envelope.
    COUNT-INVARIANT — one ``BatchUnitResult`` per Candidate; failed units are marked
    (``status == "failed"``), never dropped. The envelope->reconcile-node adapter and the
    action-decision pass are B2/B3, so this returns raw envelopes and creates no nodes.

    ``infer`` is the injected SYNC model seam (see ``_make_recall_infer``) — passed in rather
    than built here so the helper is testable with a fake infer and zero model dependency,
    mirroring extract_batch's own injected-seam design.
    """
    # Lazy import: synapt-extract is an optional dependency (the `extract` extra) since
    # SYNAPT_USE_EXTRACT is off-by-default; importing at module scope would break every
    # existing consolidate caller that never installs it. Only the flag-on path needs it.
    from synapt.extract.batch import extract_batch, BatchUnit
    from synapt.recall.identify import identify, batch_unit_id

    candidates = identify(cluster)
    if not candidates:
        return []
    # Source-date resolution anchor: each candidate's OWN journal-entry timestamp (date-only)
    # threads into BatchUnit.date, which extract_batch renders into the Stage-1 prompt so a
    # relative date ("April 30") resolves against the SOURCE year, not the extraction year —
    # recall's half of the wrong-year fix (Sentinel's real-path finding). Fail-safe to None.
    units = [
        BatchUnit(
            id=batch_unit_id(cluster_id, cand),
            text=cand.text,
            date=_candidate_source_date(cluster, cand),
        )
        for cand in candidates
    ]
    caps = _EXTRACT_CAPABILITIES if capabilities is None else capabilities
    return await extract_batch(
        units,
        infer=infer,
        produced_by=produced_by,
        capabilities=caps,
    )


def _log_extract_failure(failures_path: Path, cluster_id: str, envelope) -> bool:
    """Append one per-unit extract failure marker to ``failures_path``. Returns True when the
    marker is persisted, False when the write itself fails.

    NEVER silent-drop: every non-ok ``BatchUnitResult`` is recorded with its ``status`` +
    terminal ``reason`` and its ``source_unit_id`` (traceable to the exact journal field). If
    the marker write ITSELF fails, that is a lost failed unit — it is surfaced loudly (error
    log) and signalled to the caller (return False), never swallowed. Swallowing the write
    error would silently lose the failed unit while the cluster looks processed.
    """
    try:
        from synapt.recall._filelock import lock_exclusive, unlock
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failures_path, "a", encoding="utf-8") as ff:
            lock_exclusive(ff)
            try:
                ff.write(json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "path": "extract_batch",
                    "cluster_id": cluster_id,
                    "source_unit_id": envelope.source_unit_id,
                    "status": envelope.status,
                    "reason": envelope.reason,
                }) + "\n")
            finally:
                unlock(ff)
        return True
    except OSError as exc:
        logger.error(
            "FAILED to persist extract failure marker for unit %s (cluster %s): %s — "
            "the failed unit is NOT recorded (surfacing, not swallowing)",
            envelope.source_unit_id, cluster_id, exc,
        )
        return False


def _run_extract_path(
    cluster: list[JournalEntry],
    cluster_id: str,
    client,
    model: str,
    failures_path: Path,
    existing_nodes: list[KnowledgeNode],
    kn_path: Path,
    decision_log_path: Path | None = None,
    db=None,
    content_profile=None,
) -> ConsolidationResult | None:
    """The decomposed extract path (SYNAPT_USE_EXTRACT): B1 (extract) -> B2 (action-decision)
    -> B4 (rejoin — see ``_rejoin_create_actions``) -> B3 (reconcile — the SAME
    ``_apply_consolidation_result`` the monolithic path uses). Failed extract units are logged
    (never silent-dropped, see ``_log_extract_failure``); facts
    that extract successfully are handed to the focused action-decision pass (B2), then fed to
    reconcile exactly as the monolith's parsed LLM output would be.

    Returns a ``ConsolidationResult`` when the cluster was PROCESSED — including a zero-valued
    result when every unit failed extraction or the cluster produced no usable facts, matching
    the monolith's own "no durable patterns" outcome (a processed-but-empty cluster is not a
    failure). Returns ``None`` only on an INFRASTRUCTURE failure: an extract-inference
    exception, or a failure marker that could not be persisted — never silently reported as a
    clean (if empty) success.
    """
    infer = _make_recall_infer(client, model)
    try:
        envelopes = _run_coro_blocking(_extract_cluster_units(cluster, cluster_id, infer))
    except Exception as exc:
        logger.warning("extract-path failed for cluster %s: %s", cluster_id, exc)
        return None
    ok = [e for e in envelopes if e.status == "ok"]
    failed = [e for e in envelopes if e.status != "ok"]
    all_logged = True
    for e in failed:
        if not _log_extract_failure(failures_path, cluster_id, e):
            all_logged = False
    # A failure marker that could not be persisted is a SILENTLY-lost failed unit — the
    # never-silent-drop contract. Do not report the cluster as cleanly processed: return
    # None so the loss is visible (and the caller's retry re-attempts it), never swallowed.
    if not all_logged:
        logger.error(
            "extract-path cluster %s: one or more failure markers could not be persisted; "
            "returning failure so the lost unit is visible, not swallowed as processed",
            cluster_id,
        )
        return None
    logger.info(
        "extract-path cluster %s: %d units -> %d ok, %d failed",
        cluster_id, len(envelopes), len(ok), len(failed),
    )

    # B2: decide actions for the successfully-extracted facts against existing knowledge.
    action_items = _decide_actions(cluster, cluster_id, ok, existing_nodes, infer) if ok else []

    # B4: rejoin CREATE-bound items into compound memories before reconcile executes.
    # Shape-preserving (see _rejoin_create_actions) — B3 below needs zero changes.
    # content_profile threaded through so B4 evaluates composed content under the SAME
    # adaptive thresholds B3 will actually apply below (Sentinel, recall#884 re-review
    # round 2: this call site already receives content_profile for B3; B4 was missing it).
    action_items = _rejoin_create_actions(
        action_items, cluster_id, infer,
        decision_log_path=decision_log_path, content_profile=content_profile,
    )

    # B3: feed the SAME reconcile the monolithic path uses.
    # conflict_judge (Fix B): only the extract path wires this -- the legacy and collection
    # passes never pass it, so the escalation is always a no-op for them by construction,
    # not by a separate check (spec section 2's explicit scope boundary). UNGATED (contested-
    # memory-lifecycle reframe, spec section 10.5, Layne-ratified 2026-07-21): the judge is a
    # FLAGGER, not a resolver -- a detected conflict routes to contest+queue, never an
    # auto-applied winner, so the judge's own accuracy no longer needs gating (a false
    # CONFLICT call costs a confidence dip and a queue entry, never a lost fact). The prior
    # _gate_destructive_conflict_judgment wrapper is gone; this wires the raw judge directly.
    cluster_result = _apply_consolidation_result(
        {"nodes": action_items}, existing_nodes, cluster, kn_path,
        decision_log_path=decision_log_path, db=db, content_profile=content_profile,
        conflict_judge=_local_conflict_judge(infer),
    )
    logger.info(
        "extract-path cluster %s: reconcile -> %d created, %d corroborated, "
        "%d contradicted, %d contested",
        cluster_id, cluster_result.nodes_created, cluster_result.nodes_corroborated,
        cluster_result.nodes_contradicted, cluster_result.nodes_contested,
    )
    return cluster_result


# ---------------------------------------------------------------------------
# B2 — the action-decision pass (behind SYNAPT_USE_EXTRACT, same flag as B1)
#
# The §FINDING gap: extract_batch never sees existing knowledge and emits no action, so
# identify -> extract_batch -> reconcile with everything defaulting to "create" drops
# corroborate/contradict entirely -> node inflation (the ~70-vs-46 over-feeding dogfood).
# B2 restores the monolith's action-logic (existing-knowledge-in-context -> action +
# existing_id) as its own focused pass, batched per cluster, over B1's clean extracted facts.
#
# GATE (reframed 2026-07-14, Opus ratified): the frontier ideal (185 nodes) turned
# out to be an END-STATE node-set (no action/existing_id/contradiction fields, 0 corpus
# reversals) — NOT a per-fact action-gold, so B2 has no standalone corpus-accuracy gate the way
# identify did. This is a MECHANISM gate (synthetic existing-knowledge + facts via a fake infer,
# proving create/corroborate/contradict all fire correctly — see test_consolidate_action_-
# decision.py). Corpus-scale contradiction accuracy is descoped v1 (0 examples to validate
# against); the anti-inflation + 40-merge-group corroboration claim is measured by the Phase-C
# dogfood (B2+B3 end-state vs the 185-node ideal), not here.
#
# REVIEW FIXES (2026-07-14, Sentinel + Opus REQUEST-CHANGES, both independently fruit/code-
# confirmed): the first cut had 3 wiring gaps —
#   1. dense-cluster inflation: the per-CLUSTER action call inherited B1's flat per-UNIT
#      800-token floor, so a response covering many candidates truncated partway and every
#      unaddressed index fail-closed to "create" — the "inflation fix" silently didn't fix
#      inflation on dense clusters. Fixed: _estimate_action_decision_budget scales with
#      candidate count.
#   2. fail-closed-to-create persisted duplicates: an unaddressed/malformed index defaults to
#      create + existing_id=None with no dedup check. Fixed: _normalize_for_dedup + an exact-
#      match check against existing_nodes converts a create whose content is verbatim-
#      identical (mod case/whitespace) to corroborate — cheap, deterministic, belt-and-
#      suspenders ahead of reconcile's own fuzzy dedup, which stays untouched.
#   3. temporal_refs discarded — SUPERSEDED, see below (the temporal_refs-derivation fix and
#      its 2026-07-14 minor follow-up were both correct-in-isolation but built against a
#      hand-built envelope, not the REAL extract_batch contract; abandoned in favor of a
#      content-based per-fact approach once that was fruit-checked, see next block).
#
# TEMPORAL — ROLE, LANDED (2026-07-15, extract#31 merged to extract sprint-39): the two prior
# attempts were both duct tape — the temporal_refs-derivation cut (built against a hand-built
# envelope; role/resolved_end were unreachable at the base capability) and the B2-LLM-judges-
# temporal cut (recall re-deriving via a SECOND LLM pass what extraction already read once). The
# right fix lives in extract and now ships: extraction emits a validity ROLE (effective/expiry/
# range/superseded/point) + resolved date at the BASE temporal_refs capability, and threads each
# unit's SOURCE date as the relative-date resolution anchor (internal design spec).
# recall consumes it DETERMINISTICALLY, no LLM re-judgment: _EXTRACT_-
# CAPABILITIES requests temporal_refs; _extract_cluster_units threads BatchUnit.date
# (_candidate_source_date); _flatten_envelope_facts maps role -> valid_from/valid_until
# (_map_temporal_refs_to_bounds); _decide_actions PASSES THE BOUND THROUGH without judging it (the
# action LLM never touches temporal). A ref-less fact falls back to null — honest fallback, not the
# old unconditional placeholder. Budget (#1, _estimate_action_decision_budget) and dedup (#2,
# _normalize_for_dedup) are UNAFFECTED — independent of temporal, already fruit-confirmed by two
# reviewers, stay exactly as they are.
#
# FULL-PATH REVIEW FIXES (2026-07-15, Sentinel REQUEST CHANGES, both real, ratified by Opus after
# withdrawing his own approve on these two surfaces):
#   1. FAN-OUT FABRICATION: the first cut shared one unit-level bound onto every fact/decision the
#      unit emitted. Sentinel's real full-eb.json fruit killed the "rare" premise: 61/62 compound
#      units emit >1 output (mean 5.0) — compounds are preserved BY DESIGN so extract can atomize
#      them. Fixed in _flatten_envelope_facts: bounds SUPPRESS entirely once a unit yields >1
#      usable output — losing the one true bound beats fabricating it onto N-1 siblings.
#   2. RECONCILE SILENTLY DROPS BOUNDS ON CORROBORATE/CONTRADICT: bounds only ever reached
#      _apply_consolidation_result's create branch. The exact-match dedup guard MECHANICALLY
#      converts many creates to corroborate, so a real temporal fact routinely lost its bound one
#      layer past where the B2 tests stopped looking (dict, not persisted node). Fixed: corroborate
#      fills only MISSING persisted bounds (never overwrites a conflicting one); contradict carries
#      candidate bounds onto the replacement node AND the queued-contradiction payload — which
#      required a pending_contradictions schema migration (storage.py) and a fix to
#      server.py's _apply_supersession (previously hardcoded valid_from=now/valid_until=None on
#      confirm), so the round trip survives to the actually-persisted node, not just the queue row.
# ---------------------------------------------------------------------------

ACTION_DECISION_PROMPT = """\
You are deciding how new facts relate to existing knowledge.

## Existing Knowledge
{existing_knowledge}

## New Facts (indexed)
{candidates}

## Task
For EACH new fact above, decide exactly one action:
- "create": this is genuinely new information, not already captured above.
- "corroborate": this confirms/reinforces an EXISTING node — give its exact [id] from above.
- "contradict": this REVERSES or invalidates an EXISTING node's claim — give its exact [id] \
and a one-sentence contradiction_note explaining the reversal.

Rules:
1. Only use "corroborate" or "contradict" when you can cite the EXACT existing node id shown \
in brackets above. Never invent an id.
2. Default to "create" when uncertain.
3. contradiction_note is required for "contradict", omit or leave empty otherwise.

Output ONLY valid JSON, no markdown fences, no explanation. One action object PER fact index, \
matched by "index" (not by list position):
{{"actions": [{{"index": 0, "action": "create", "existing_id": null, "contradiction_note": ""}}, \
{{"index": 1, "action": "corroborate", "existing_id": "kn_abc123", "contradiction_note": ""}}]}}
"""

_VALID_ACTIONS = ("create", "corroborate", "contradict")


# Role -> bound routing. Direction rides in extract's ROLE field (extract#31); recall never
# re-judges it. ``point`` defaults to a start (a bare instant with no start/end direction).
_ROLE_TO_VALID_FROM = frozenset({"effective", "point"})
_ROLE_TO_VALID_UNTIL = frozenset({"expiry", "superseded"})


def _map_temporal_refs_to_bounds(temporal_refs) -> tuple[str | None, str | None]:
    """Map a unit's extract_batch ``temporal_refs`` to ``(valid_from, valid_until)`` by ROLE,
    DETERMINISTICALLY — the recall-side replacement for both the abandoned second-LLM judgment
    and the held null placeholder (module TEMPORAL — ROLE note).

    Role -> bound: ``effective`` -> valid_from; ``expiry``/``superseded`` -> valid_until;
    ``range`` -> both (resolved, resolved_end); ``point`` -> valid_from. A ref with no role (or
    an unknown one) cannot encode direction and contributes nothing — null is the honest
    FALLBACK, now scoped to exactly that case rather than unconditional. First-non-null-wins per
    bound in list order (deterministic; no min/max heuristic that could reorder under ties).
    Dates pass ``_validate_iso_date`` — a malformed date yields no bound. Non-dict refs are
    skipped so a single bad element never crashes a cluster's consolidation."""
    valid_from: str | None = None
    valid_until: str | None = None
    for ref in temporal_refs or []:
        if not isinstance(ref, dict):
            continue
        role = ref.get("role")
        resolved = _validate_iso_date(ref.get("resolved"))
        if role in _ROLE_TO_VALID_FROM:
            valid_from = valid_from or resolved
        elif role in _ROLE_TO_VALID_UNTIL:
            valid_until = valid_until or resolved
        elif role == "range":
            valid_from = valid_from or resolved
            valid_until = valid_until or _validate_iso_date(ref.get("resolved_end"))
        # role absent/unknown -> no contribution (direction is unknowable)
    return valid_from, valid_until


def _flatten_envelope_facts(envelopes) -> list[dict]:
    """Flatten B1's per-unit envelopes into a flat, ordered list of candidate facts for the
    action-decision pass. Only ``status == "ok"`` envelopes contribute — failed units were
    already logged by B1 (never silently re-surfaced here). Both ``facts[]`` (tagged with their
    own ``category``, default "fact") and ``decisions[]`` (tagged "decision") flow through, each
    carrying its source envelope's ``source_unit_id`` for traceability (not threaded onto the
    KnowledgeNode itself — see ``_decide_actions``'s ``source_turns`` note).

    TEMPORAL: each fact/decision carries the ``valid_from``/``valid_until`` mapped
    DETERMINISTICALLY from its unit's ``temporal_refs`` by role (``_map_temporal_refs_to_bounds``,
    module TEMPORAL — ROLE note). ``temporal_refs`` are extraction(unit)-level (siblings of
    ``facts[]``, confirmed against real extract_batch output) — the IL has NO fact<->ref
    attribution, so assigning a ref to one output out of several is a guess, not a derivation.

    FAN-OUT SUPPRESSION (Sentinel, 2026-07-15, real full-eb.json fruit: of 62 OK compound units,
    61 emitted >1 durable output, mean 5.0 — compounds are preserved BY DESIGN precisely so
    extract can atomize them, so multi-output units are the NORM, not the rare case the original
    bound-sharing design assumed): bounds are computed per envelope ONLY when that unit yields
    exactly one usable output (facts + decisions combined). Once a unit yields >1 output, its
    bound is SUPPRESSED entirely (null for every output from that unit) — sharing would fabricate
    a boundary on N-1 unrelated memories; losing the one true bound is the safer failure. No LLM
    re-judgment either way; a ref-less or suppressed unit falls back to null bounds.
    """
    out: list[dict] = []
    for env in envelopes:
        if env.status != "ok" or not env.extraction:
            continue
        extraction = env.extraction
        # Usable outputs FIRST — the suppression decision needs the TRUE combined count before
        # any bound is computed, so it can't be decided while iterating facts alone.
        unit_items: list[dict] = []
        for fact in extraction.get("facts") or []:
            text = fact.get("text") if isinstance(fact, dict) else None
            if text:
                unit_items.append({"text": text, "category": fact.get("category") or "fact"})
        for dec in extraction.get("decisions") or []:
            text = dec.get("text") if isinstance(dec, dict) else None
            if text:
                unit_items.append({"text": text, "category": "decision"})

        if len(unit_items) > 1:
            valid_from, valid_until = None, None  # fan-out: suppress, never fabricate
        else:
            valid_from, valid_until = _map_temporal_refs_to_bounds(extraction.get("temporal_refs"))

        for item in unit_items:
            out.append({
                "text": item["text"],
                "category": item["category"],
                "source_unit_id": env.source_unit_id,
                "valid_from": valid_from,
                "valid_until": valid_until,
            })
    return out


def _normalize_for_dedup(text: str) -> str:
    """Lowercase + collapse whitespace, for a cheap EXACT-match dedup check — NOT the fuzzy
    Jaccard/embedding dedup reconcile's own create-path already does downstream (that stays
    untouched). This catches the specific failure mode a fail-closed-to-create fact's content
    is verbatim-identical (modulo case/whitespace) to something already persisted, without
    needing embeddings or a similarity threshold."""
    return " ".join(text.lower().split())


# ~4 chars/token (the same approximation _estimate_response_budget already uses); a per-fact
# action item ({"index":N,"action":"corroborate","existing_id":"kn_xxxxxxxx","contradiction_
# note":"..."}) runs ~60-80 tokens generously — 60 leaves headroom without being wasteful.
_ACTION_ITEM_TOKEN_ESTIMATE = 60


def _estimate_action_decision_budget(prompt: str, n_facts: int) -> int:
    """Estimate ``max_tokens`` for B2's action-decision response, scaled to candidate count.

    ``_estimate_response_budget`` (built for the monolith's fixed-shape cluster-summary output)
    SHRINKS its allowance as the prompt grows — the wrong direction for B2: its per-cluster
    batched call can address up to ~50-65 candidates (a full cluster's prefiltered facts), and
    its OUTPUT size scales with candidate count, not just prompt size. The flat
    ``MIN_RESPONSE_TOKENS`` floor silently truncated a dense cluster's response around fact
    15-20; every unaddressed index then fail-closed to "create" — the exact dense-cluster
    inflation Sentinel's fruit-check and Opus's independent code-read both caught (the
    "inflation fix" wasn't fixing inflation on dense clusters). Takes whichever is LARGER of a
    fact-count-scaled estimate and the monolith's own context-aware estimate — generous is
    free ("no upper cap, the model stops at EOS naturally" per ``_estimate_response_budget``'s
    own rationale); only a too-small floor was ever the actual bug.
    """
    scaled = MIN_RESPONSE_TOKENS + _ACTION_ITEM_TOKEN_ESTIMATE * n_facts
    return max(scaled, _estimate_response_budget(prompt))


def _format_facts_for_action_decision(facts: list[dict]) -> str:
    """Render flattened facts as an indexed list for the action-decision prompt. The index is
    the matching key the model must echo back (robust to a batched call reordering or only
    partially addressing the facts — see ``_parse_action_decision_response``)."""
    return "\n".join(f"[{i}] ({f['category']}) {f['text']}" for i, f in enumerate(facts))


def _build_action_decision_prompt(
    facts: list[dict],
    existing_nodes: list[KnowledgeNode],
    cluster: list[JournalEntry],
) -> str:
    """Build the action-decision prompt: the SAME existing-knowledge formatting the monolith
    prompt + reconcile already use (``_format_existing_knowledge``, reused not rebuilt), plus
    the indexed candidate facts."""
    existing = _format_existing_knowledge(existing_nodes, cluster=cluster)
    candidates = _format_facts_for_action_decision(facts)
    return ACTION_DECISION_PROMPT.format(existing_knowledge=existing, candidates=candidates)


def _parse_action_decision_response(response: str) -> dict[int, dict] | None:
    """Parse the action-decision response into an ``index -> action-item`` dict, or None if the
    response is unparseable / not the expected shape. A malformed individual entry (non-dict,
    missing/non-int index) is skipped rather than failing the whole batch — the caller degrades
    any un-matched fact index to ``action="create"`` (fail-closed, never dropped)."""
    parsed = _parse_llm_response(response)
    if not parsed:
        return None
    raw_actions = parsed.get("actions")
    if not isinstance(raw_actions, list):
        return None
    by_index: dict[int, dict] = {}
    for item in raw_actions:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if not isinstance(idx, int):
            continue
        by_index[idx] = item
    return by_index


def _decide_actions(
    cluster: list[JournalEntry],
    cluster_id: str,
    ok_envelopes,
    existing_nodes: list[KnowledgeNode],
    infer,
) -> list[dict]:
    """B2: the action-decision pass. Flattens B1's successfully-extracted facts, asks a focused
    LLM pass (existing knowledge + new facts -> per-fact action) and returns B4-ready dicts:
    EXACTLY ``_apply_consolidation_result``'s ``parsed["nodes"]`` item shape (``action``,
    ``existing_id``, ``content``, ``category``, ``tags``, ``source_turns``,
    ``contradiction_note``, ``valid_from``, ``valid_until``), PLUS ``source_unit_id`` — a
    pass-through B4 needs for decision-log provenance (guard 3, contract-answer
    2026-07-15); reconcile ignores the extra key (verified, not assumed — B3 stays
    zero-changes). ``confidence`` is NOT included — reconcile computes it itself (B1
    contract-read Finding 3) with no fallback gap.

    ``valid_from``/``valid_until`` come from each fact's unit-level ``temporal_refs``, mapped
    DETERMINISTICALLY by role in ``_flatten_envelope_facts`` (module TEMPORAL — ROLE note) — this
    pass PASSES THEM THROUGH, it does not judge them. Direction is read once at extraction
    (extract#31's role field), never re-derived by a second recall-side LLM pass. A ref-less fact
    carries null (the honest fallback), and a garbage action response fail-closes the ACTION to
    create while the bound rides through unchanged — temporal is decoupled from the action.

    FAIL-CLOSED, never drops a fact: an unparseable/malformed response, an infer exception, an
    invalid action value, or any fact index the response didn't address all degrade that fact's
    ACTION to "create" (mirrors the monolith's own existing_id-not-found -> create fallback) —
    the fact itself is always represented in the output (count-invariant, matching B1's
    discipline at this layer). A create-defaulted (or model-CHOSEN-create) fact whose content
    EXACTLY matches (``_normalize_for_dedup``) an already-existing node is converted to
    corroborate that node instead — never-drop must not mean never-dedup; see the REVIEW FIXES
    module note (blocker #2, the persisted-duplicate finding).

    ``infer`` is the SAME pluggable sync seam B1 uses (``_make_recall_infer``) — one call per
    cluster (a cluster's facts + its existing-knowledge set are bounded, so batching per cluster
    is v1; see the B2 spec for the per-fact-vs-batched tradeoff). The request's ``max_tokens``
    is scaled to candidate count (``_estimate_action_decision_budget``, blocker #1) — a flat
    per-unit floor silently truncated dense clusters and mass-triggered the fail-closed path.

    ``source_turns`` is intentionally always ``[]`` here — see module note: the deterministic
    prefilter's ``Candidate.attr`` (session_id/entry_index/field/index) is NOT transcript-turn-
    shaped, and populating a KnowledgeNode's ``source_turns`` with a non-turn value would make
    ``resolve_source_turns()`` SKIP real enrichment for that node forever (its early-exit is
    ``if node.source_turns: continue``, verified at consolidate.py) — worse than leaving it
    empty for the SAME post-processing resolver that already backfills genuine transcript
    matches for monolith-path nodes when the LLM didn't supply turns.
    """
    facts = _flatten_envelope_facts(ok_envelopes)
    if not facts:
        return []

    prompt = _build_action_decision_prompt(facts, existing_nodes, cluster)
    budget = _estimate_action_decision_budget(prompt, len(facts))
    request = {
        "prompt": prompt,
        "messages": [{"role": "user", "content": prompt}],
        "capabilities": [],
        "max_tokens": budget,
    }
    try:
        response = infer(request)
    except Exception as exc:
        logger.warning("action-decision inference failed for cluster %s: %s", cluster_id, exc)
        response = None

    by_index = _parse_action_decision_response(response) if response else None
    if by_index is None:
        by_index = {}

    # Exact-match dedup ahead of reconcile: see _normalize_for_dedup. Built once per cluster,
    # not per-fact, since existing_nodes is fixed for the whole call.
    existing_by_normalized = {
        _normalize_for_dedup(n.content): n.id for n in existing_nodes
    }

    results: list[dict] = []
    for i, fact in enumerate(facts):
        item = by_index.get(i, {})
        action = item.get("action")
        if action not in _VALID_ACTIONS:
            action = "create"
        existing_id = item.get("existing_id")
        tags = item.get("tags")
        if not isinstance(tags, list):
            tags = []
        contradiction_note = item.get("contradiction_note")
        if not isinstance(contradiction_note, str):
            contradiction_note = ""
        if action == "create":
            matched_id = existing_by_normalized.get(_normalize_for_dedup(fact["text"]))
            if matched_id:
                action = "corroborate"
                existing_id = matched_id
        results.append({
            "action": action,
            "existing_id": existing_id,
            "content": fact["text"],
            "category": fact["category"],
            "tags": tags,
            "source_turns": [],
            "contradiction_note": contradiction_note,
            # Temporal bounds are mapped from the ENVELOPE's role in _flatten_envelope_facts
            # (module TEMPORAL — ROLE note), NOT judged by this action pass — a ref-less fact
            # carries null (honest fallback), and the action LLM's response never supplies these
            # (it is not even asked to; see the action-decision prompt). Decoupling the bound from
            # the action means it survives the fail-closed-to-create path unchanged.
            "valid_from": fact.get("valid_from"),
            "valid_until": fact.get("valid_until"),
            # Pass-through provenance for B4 (internal design spec, guard 3,
            # contract-answer 2026-07-15): reconcile ignores unknown dict keys (verified,
            # not assumed — every field _apply_consolidation_result reads comes off .get(), never
            # **kwargs/dataclass unpacking), so this rides through B3 unused until B4 unions member
            # ids into the decision log. Not a KnowledgeNode field — no schema/storage change.
            "source_unit_id": fact.get("source_unit_id"),
        })
    return results


# ---------------------------------------------------------------------------
# B4 — the rejoin/compose stage (behind SYNAPT_USE_EXTRACT, same flag as B1/B2)
#
# Position: B2 -> B4 -> B3. Every stage up to here atomizes or passes atoms through; nothing
# ever reassembles. The Phase C dogfood (internal design spec)
# found the decomposed path finds more and lies less than legacy but fragments compound
# memories (378 nodes vs the 185-node frontier ideal, boundary F1 9.1% vs legacy's 23.0%) —
# and fragmentation CORRUPTS truth (3 concrete false memories in verification). B4 rejoins:
# it takes a cluster's CREATE-bound action items and composes them into compound memories at
# ideal granularity before reconcile executes. Only CREATE items participate — corroborate/
# contradict target existing nodes and carry no new granularity, so they pass through B4
# untouched and never reach the model.
#
# Five contract guards, each its own mechanism-TDD test (recall#883, Sentinel):
#   1. COUNT-INVARIANT, never-drop: every create index appears in exactly one output group.
#      A group with ANY invalid/hallucinated index is dropped WHOLE (per-group degradation —
#      the composed text was written describing a membership set that doesn't exist once one
#      member is fake; trusting it for the survivors would persist text written for a
#      different group). Duplicate membership ACROSS groups invalidates the ENTIRE response
#      (whole-cluster fail-open) — a stronger violation than a single bad group, since it
#      means the model's own bookkeeping is incoherent, not just one group's target set.
#   2. COMPOSE FROM MEMBERS ONLY: v1 enforcement is PROMPT-LEVEL (the composition prompt pins
#      anti-corruption instructions against the three observed truth-corruption classes —
#      negation flip, convention misread, relation/causal misread). Runtime entailment
#      checking is v2, gated on the dogfood's source-clean precision axis.
#   3. METADATA UNION: tags unioned, category by majority-of-members (ties broken by the
#      FIRST member in the MODEL's own group order — Counter.most_common()'s stable sort
#      naturally does this when members are visited in group order). Durable NODE-level
#      provenance is a v2 follow-up; v1 traceability lives in the decision log (member
#      source_unit_ids + content digests + composed content), keeping B3/KnowledgeNode
#      unchanged.
#   4. TEMPORAL CONFLICT RULE: identical bounds across bounded members -> carry; conflicting
#      non-null bounds (including two members with different single-sided bounds that would
#      otherwise synthesize an unasserted range) -> the group is REJECTED and its members
#      revert to individual pass-through (per-group degradation, same mechanism as #1's
#      invalid-index drop — NOT the whole-cluster fail-open of #1's duplicate case); exactly
#      one bounded member -> carry that member's bound verbatim (safe because #878's fan-out
#      suppression already made "non-null bound => sole output of its source unit" true at
#      write time — no text re-detection needed here).
#   5. FAIL-OPEN degradation: an unparseable/wrong-shape response or an infer exception
#      degrades the WHOLE cluster's creates to pass-through, never blocked, never dropped —
#      plus a loud, non-silent B4_COMPOSE_FAIL_OPEN warning naming the cluster and reason.
# ---------------------------------------------------------------------------

_B4_COMPOSE_PROMPT = """\
You are grouping and composing related facts into compound memories.

## Facts to consider (indexed)
{candidates}

## Task
Decide which of the facts above describe the SAME underlying memory and should be composed \
together into one compound entry. A fact that stands alone should be its own singleton group.

For EACH fact index above, put it in exactly one group. Give each group's member indices and \
ONE composed content string for that group.

Also decide, for each group, whether it "supersedes" any OTHER index — meaning that OTHER \
index describes the SAME subject at an earlier point in time, and this group's content \
updates, corrects, or contradicts it (not merely relates to it or shares its vocabulary). \
List those other indices in a "supersedes" array (empty if none). Two facts about the same \
subject that are simply complementary, not in conflict, are NOT a supersession — shared \
vocabulary alone is never enough; only flag a genuine same-subject update, correction, or \
contradiction.

Rules:
1. Compose ONLY from the given MEMBER facts — never assert a relation, cause, or qualifier \
that is not directly present in a member's own text. Do not flip a negation (a fact stating \
something is unrelated must stay unrelated in the composition). Do not alter a stated \
convention or definition (e.g. what a label means) when restating it.
2. Every index above must appear in EXACTLY ONE group.
3. Output ONLY valid JSON, no markdown fences, no explanation.

{{"groups": [{{"indices": [0, 1], "content": "...", "supersedes": []}}, \
{{"indices": [2], "content": "...", "supersedes": [0]}}]}}
"""


def _build_rejoin_prompt(creates: list[dict]) -> str:
    """Render the create-only candidates as an indexed list for the compose prompt. Indices
    are positions within *creates* (the FILTERED create-only list the model sees) — NOT
    positions in the full action_items list, since the model never sees non-create items."""
    candidates = "\n".join(f"[{i}] {item['content']}" for i, item in enumerate(creates))
    return _B4_COMPOSE_PROMPT.format(candidates=candidates)


# A composed group's content is a full merged sentence (potentially several member facts
# folded together) — larger than a single action-decision item, so a bigger per-group
# estimate than B2's _ACTION_ITEM_TOKEN_ESTIMATE. Mirrors _estimate_action_decision_budget's
# shape: do NOT repeat the flat-800-floor bug class at this stage either.
_B4_COMPOSE_GROUP_TOKEN_ESTIMATE = 150


def _estimate_rejoin_budget(prompt: str, n_creates: int) -> int:
    """Estimate ``max_tokens`` for B4's compose response, scaled to create-item count (see
    module note — the same fix class as ``_estimate_action_decision_budget``).

    Deliberately does NOT take ``max(..., _estimate_response_budget(prompt))`` the way B2
    does: B4's prompt itself grows with ``n_creates`` (more candidates listed), so
    ``_estimate_response_budget``'s CONTEXT_BUDGET-minus-prompt-tokens term actively SHRINKS
    as the cluster gets denser — the opposite of what "scale with create count" needs. A
    pure count-scaled floor is generous on its own (no upper cap; the model stops at EOS)
    and, unlike the combinator, never fights its own growth as the prompt lengthens.
    """
    return MIN_RESPONSE_TOKENS + _B4_COMPOSE_GROUP_TOKEN_ESTIMATE * n_creates


def _group_temporal_bound(members: list[dict]) -> tuple[str | None, str | None] | None:
    """Guard 4: decide the (valid_from, valid_until) a composed group should carry, or None
    if the group must be REJECTED for conflicting bounds (caller then reverts its members to
    individual pass-through — see module note, per-group degradation).

    A member "has a bound" when either field is non-null. Zero bounded members -> (None,
    None), trivially. Exactly one bounded member -> carry its tuple verbatim (safe per #878's
    fan-out suppression: a non-null bound already means "sole output of its unit"). Two or
    more bounded members whose tuples are ALL identical -> carry that shared tuple. Any
    difference between bounded members' tuples — including two members with different
    single-sided bounds that would otherwise synthesize an unasserted range — is a conflict:
    reject the group. Precision-first: a validity boundary is real evidence of two different
    memories; do not merge across it.
    """
    bounded = [
        (m.get("valid_from"), m.get("valid_until"))
        for m in members
        if m.get("valid_from") or m.get("valid_until")
    ]
    if not bounded:
        return (None, None)
    distinct = set(bounded)
    if len(distinct) == 1:
        return bounded[0]
    return None


# The composed content's own safety ceiling, matched to _apply_consolidation_result's real
# create-branch pipeline (content = scrub_text(_tw(str(raw_node.get("content", "")), 300))) —
# NOT a new independent limit. A composition that would arrive at B3 empty-after-scrub (e.g.
# whitespace-only) is silently DROPPED there (0 nodes created, both members lost); one that
# exceeds 300 chars is silently WORD-TRUNCATED (the load-bearing final clause can vanish with
# no warning). Both are real, fruit-confirmed (Sentinel, recall#884 re-review): B4 must reject
# a group whose composed content would hit either path, BEFORE it ever reaches B3, and fall
# the group's members back to individual pass-through instead.
_B4_COMPOSE_CONTENT_MAX_CHARS = 300


def _b4_composed_content_is_safe(content: str, content_profile=None) -> bool:
    """Whether *content* is safe to persist as a B4-composed node.

    Two parts. First, ONE check B4 alone needs, stricter than B3 itself: reject (don't
    silently truncate) raw content over B3's real 300-char ceiling. B3 tolerates truncation —
    ``_tw`` silently cuts at 300 chars and moves on, the monolith's long-standing behavior for
    ANY create-action content. B4 does not: a COMPOSED sentence that needs truncation has
    already lost information the model was told to preserve faithfully (guard 2), and unlike
    an atomic fact, B4 has a safe fallback B3 doesn't — degrade the whole group back to its
    original, already-safe, un-composed members — so it refuses rather than risk a silently
    cut load-bearing clause (Sentinel, recall#884 re-review, original finding).

    Second, everything else routes through ``_evaluate_create_content`` — the SAME function
    B3's create branch calls — so this verdict is provably identical to what B3 will compute
    later: parity by construction (Sentinel, recall#884 re-review round 2: an earlier version
    of this function hard-coded ``threshold=120``/``content_type=None``/no-normalize and
    diverged from what B3 actually evaluates once ``content_profile`` flows in, which it
    already does at B3 via ``_run_extract_path`` — fruit: a real mixed-profile threshold=200
    dropped a 144-char composition B4 had waved through; a 152-char section-prefixed
    composition B4 waved through stripped to a 118-char low-specificity remainder at B3).
    """
    if len(content) > _B4_COMPOSE_CONTENT_MAX_CHARS:
        return False
    return _evaluate_create_content(content, content_profile) is not None


def _log_b4_compose_decision(
    decision_log_path: Path,
    cluster_id: str,
    member_indices: list[int],
    members: list[dict],
    composed_content: str,
) -> bool:
    """Durable v1 traceability for a B4 composition (guard 3): the full member -> group
    mapping in the decision log — the existing durable audit substrate — rather than a new
    KnowledgeNode field (B3/schema stay unchanged).

    Returns whether the entry was actually persisted. UNLIKE ``_log_dedup_decision`` (whose
    write failure is harmless best-effort telemetry), guard 3 chose the decision log as v1's
    ONLY durable member provenance — best-effort-swallow-the-OSError would silently persist an
    untraceable compound with no record anywhere of what it was composed from (fruit-confirmed,
    Sentinel, recall#884 re-review). The caller must reject the composition on ``False``, not
    treat this as fire-and-forget logging.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "b4-compose",
        "cluster_id": cluster_id,
        "member_indices": list(member_indices),
        "source_unit_ids": [m.get("source_unit_id") for m in members],
        "member_content_digests": [
            hashlib.sha256(m["content"].encode("utf-8")).hexdigest() for m in members
        ],
        "composed_content": composed_content,
    }
    try:
        from synapt.recall._filelock import lock_exclusive
        decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(decision_log_path, "a", encoding="utf-8") as f:
            lock_exclusive(f)
            f.write(json.dumps(entry) + "\n")
            f.flush()
        return True
    except OSError:
        logger.debug("Failed to write B4 compose decision log")
        return False


def _b4_fail_open(action_items: list[dict], cluster_id: str, reason: str) -> list[dict]:
    """Guard 5: degrade the WHOLE cluster's creates to pass-through, loudly. Never blocked,
    never dropped — worst case is the status quo ante (every create stays atomic)."""
    logger.warning(
        "B4_COMPOSE_FAIL_OPEN cluster=%s reason=%s", cluster_id, reason,
    )
    return action_items


# B4 compose: bounded retry + rejection telemetry (A4, grouping-contract tighten, sprint-41).
# Not a new numbered guard — guard-6 is reserved for A3's supersession guard (A1 contract,
# config f552875 section 4). This is a tightening of guards 1/3/4's existing failure paths: a
# bounded retry on the response-level validation those guards already perform, plus loud
# telemetry on the per-group rejections guard 4 and the content-safety check already make.
#
# The 2026-07-15 dogfood packet's 3 whole-cluster fail-opens were ALL caused by a response
# failing guard-3 schema validation or the duplicate-membership scan — never by an infer-
# backend exception. A single bounded retry with a prompt naming exactly what was wrong gives
# the model one real chance to self-correct before the cluster degrades to full atomization.
# Deliberately does NOT retry a well-formed response that a later PER-GROUP guard rejects
# (content-unsafe / temporal-conflict) — those are correct, deliberate degradations of valid
# model output, not malformed output worth re-asking for.
_B4_COMPOSE_MAX_ATTEMPTS = 2  # 1 initial + 1 corrective retry — bounded, never unbounded


def _b4_validate_compose_response(
    parsed: dict, n: int,
) -> (
    "tuple[list[tuple[list, str, list, str | None]], list[list[int]], None] "
    "| tuple[None, None, str]"
):
    """Guard 3 (schema) + duplicate-membership validation for ONE compose attempt.

    Returns ``(raw_groups, group_real_members, None)`` on success or ``(None, None, reason)``
    on failure. ``reason`` feeds both the eventual ``B4_COMPOSE_FAIL_OPEN`` marker (attempts
    exhausted) and the next attempt's corrective prompt (attempts remain) — see the bounded-
    retry module note above.

    Each ``raw_groups`` entry is ``(indices, content, supersedes, malformed_supersedes_reason)``.
    ``supersedes`` (guard 6, A3) is OPTIONAL; the base composition contract (indices/content) is
    guard 3's concern, and a broken supersession hint on top of an otherwise-valid group is
    never the same class of failure as a broken group itself. Malformed handling follows
    Sentinel's spec-author ruling (A3 re-review, m_4adb43c0):

    - A MISSING ``supersedes`` field degrades silently to no claim — backward compatible, no
      telemetry, since the field didn't exist before this sprint.
    - A PRESENT-but-wrong-typed field, or a list containing invalid/out-of-range targets, is a
      real detection-signal failure, not a benign absence: ``malformed_supersedes_reason`` names
      why, and the caller emits exactly one ``b4-group-rejected`` telemetry entry (stage
      ``malformed-supersedes``) per such group — "malformed optional detection metadata must
      never cost data, but silently dropping it erases the model-quality signal A6 needs."
    - Valid targets inside an otherwise-partially-invalid list are KEPT, not discarded
      wholesale, matching that same "never cost data" rule.
    """
    raw_groups: list[tuple[list, str, list, "str | None"]] = []
    for g in parsed["groups"]:
        if not isinstance(g, dict):
            return None, None, (
                "compose response contains a structurally malformed group entry (not an object)"
            )
        idxs = g.get("indices")
        content = g.get("content")
        if not isinstance(idxs, list) or not isinstance(content, str):
            return None, None, (
                "compose response contains a structurally malformed group entry "
                "(indices/content wrong type — schema violation)"
            )

        malformed_reason: "str | None" = None
        if "supersedes" not in g:
            supersedes: list[int] = []
        else:
            raw_supersedes = g["supersedes"]
            if not isinstance(raw_supersedes, list):
                supersedes = []
                malformed_reason = (
                    "supersedes field was present but not a list — supersession signal "
                    "ignored, composition preserved"
                )
            else:
                supersedes = [
                    i for i in raw_supersedes
                    if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n
                ]
                if len(supersedes) != len(raw_supersedes):
                    malformed_reason = (
                        "supersedes list contained invalid or out-of-range targets — "
                        "filtered, valid targets retained, composition preserved"
                    )
        raw_groups.append((idxs, content, supersedes, malformed_reason))

    group_real_members: list[list[int]] = []
    seen: set[int] = set()
    duplicate = False
    for idxs, _content, _supersedes, _malformed in raw_groups:
        real = [
            i for i in idxs
            if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n
        ]
        group_real_members.append(real)
        for i in real:
            if i in seen:
                duplicate = True
            seen.add(i)
    if duplicate:
        return None, None, "duplicate membership across groups"

    return raw_groups, group_real_members, None


def _log_b4_group_rejection(
    decision_log_path: Path | None,
    cluster_id: str,
    member_indices: list[int],
    stage: str,
    reason: str,
) -> None:
    """B4 compose rejection telemetry (A4): best-effort per-group rejection telemetry,
    distinguishable by ``stage``.

    Closes the exact gap the 2026-07-15 dogfood packet named (B4-FRUIT.md): "production
    telemetry exposes that these groups degraded to member pass-through, but it does not
    record which per-group guard rejected each proposal... A content-versus-temporal
    breakdown therefore cannot be claimed from this run." Unlike guard 3's compose-SUCCESS
    provenance (whose loss means an untraceable persist — reject the composition), a lost
    REJECTION-telemetry entry loses nothing that was ever persisted, so this is pure
    best-effort logging, same shape as ``_log_dedup_decision``: never disrupts consolidation.
    """
    logger.info(
        "B4_GROUP_REJECTED cluster=%s stage=%s reason=%s member_indices=%s",
        cluster_id, stage, reason, member_indices,
    )
    if decision_log_path is None:
        return
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "b4-group-rejected",
        "cluster_id": cluster_id,
        "member_indices": list(member_indices),
        "stage": stage,
        "reason": reason,
    }
    try:
        from synapt.recall._filelock import lock_exclusive
        decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(decision_log_path, "a", encoding="utf-8") as f:
            lock_exclusive(f)
            f.write(json.dumps(entry) + "\n")
            f.flush()
    except OSError:
        logger.debug("Failed to write B4 group-rejection decision log")


def _entry_index_if_parseable(source_unit_id: "str | None") -> "int | None":
    """Parse the real ``{cluster}:{entry_index}:{field}:{item}`` format's ``entry_index``
    numerically when the shape matches (exactly 4 colon-segments, second segment a plain int).
    Returns ``None`` for any other shape — an honest degradation the caller falls back on,
    never a crash on an unexpected ``source_unit_id``."""
    parts = (source_unit_id or "").split(":")
    if len(parts) != 4:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _source_unit_id_order(a: "str | None", b: "str | None") -> int:
    """Chronological order of *a* relative to *b*, for guard 6's direction resolution: ``-1`` if
    *a* is earlier, ``1`` if *b* is earlier, ``0`` if they are TIED — same ``entry_index`` (or,
    for a non-real-format shape, byte-identical strings).

    A tie is a real, distinct outcome, not a coin flip resolved by whichever side the boolean
    comparison's ``else`` branch happens to favor (Sentinel, A3 re-review — the original
    boolean-only ``_source_unit_id_is_earlier`` returned ``False`` for BOTH ``is_earlier(a, b)``
    and ``is_earlier(b, a)`` on a tie, and the caller's if/else treated "not earlier" as
    "definitely later," silently suppressing whichever side happened to be the loop's `target`.
    A1 §7 explicitly names same-entry siblings — the tie is on ``entry_index`` itself, which
    spans across fields (a ``decisions`` item and a ``done`` item from the same entry tie just
    as much as two items from the same field) — as "no usable direction signal from
    source_unit_id alone"; the caller must preserve both sides on ``0``, never suppress).

    Parses the real 4-segment format's ``entry_index`` numerically when BOTH sides match it —
    robust to any future ``_split_large_cluster`` ``max_size`` change; there is no hidden
    lexicographic-vs-numeric equivalence here to silently invert if that bound ever climbs past
    9 (Opus, 2026-07-16). Falls back to full-string comparison for any other shape, which is
    exactly what A2's synthetic test fixtures use (e.g. ``"s020c00:0"`` — 2 segments, not 4).
    """
    a_idx = _entry_index_if_parseable(a)
    b_idx = _entry_index_if_parseable(b)
    if a_idx is not None and b_idx is not None:
        if a_idx == b_idx:
            return 0
        return -1 if a_idx < b_idx else 1
    a_str, b_str = (a or ""), (b or "")
    if a_str == b_str:
        return 0
    return -1 if a_str < b_str else 1


def _log_b4_supersede_decision(
    decision_log_path: "Path | None",
    cluster_id: str,
    superseded_index: int,
    superseding_index: int,
    superseded_source_unit_id: "str | None",
    superseding_source_unit_id: "str | None",
    superseded_content: str,
    superseding_content: str,
) -> bool:
    """Guard 6 (A3, supersession suppression): durable traceability for a suppressed same-batch
    supersession — and REJECT-ON-FAILURE, not fire-and-forget (Sentinel, A3 re-review:
    "suppression is only truth-preserving when the audit record is durable" — a lost
    supersede-telemetry entry means the suppressed candidate is gone with zero trace of why).
    Returns whether the entry was actually persisted; the caller must NOT add the superseded
    position to ``suppressed_positions`` unless this returns ``True`` — on any failure to
    persist, both candidates stay in output instead.

    ``decision_log_path is None`` ALSO returns ``False`` here (Sentinel, A3 audit clarification)
    — guard 3's compose path treats a ``None`` path as a legitimate silent opt-out, but not
    because composition merely "adds a node" (it can replace several member creates with one
    compound, same as here). The true boundary: compose-from-members PRESERVES every member's
    content by contract — the atoms are still there, just folded together — while supersession
    INTENTIONALLY DISCARDS one claim entirely. A skipped compose-provenance breadcrumb loses a
    trace of how content was folded; a skipped supersede breadcrumb loses the only record that a
    claim ever existed before it vanished. That is the same risk class as a write failure, not a
    lesser one. Every real production call site (``_run_extract_path``) always supplies a
    concrete path regardless."""
    if decision_log_path is None:
        return False
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "b4-supersede",
        "cluster_id": cluster_id,
        "superseded_index": superseded_index,
        "superseding_index": superseding_index,
        "superseded_source_unit_id": superseded_source_unit_id,
        "superseding_source_unit_id": superseding_source_unit_id,
        "superseded_content_digest": hashlib.sha256(
            superseded_content.encode("utf-8")
        ).hexdigest(),
        "superseding_content_digest": hashlib.sha256(
            superseding_content.encode("utf-8")
        ).hexdigest(),
    }
    try:
        from synapt.recall._filelock import lock_exclusive
        decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(decision_log_path, "a", encoding="utf-8") as f:
            lock_exclusive(f)
            f.write(json.dumps(entry) + "\n")
            f.flush()
        return True
    except OSError:
        logger.warning(
            "B4_SUPERSEDE_LOG_FAILED cluster=%s superseded_index=%s — "
            "suppression NOT applied, both candidates retained",
            cluster_id, superseded_index,
        )
        return False


def _rejoin_create_actions(
    action_items: list[dict],
    cluster_id: str,
    infer,
    *,
    decision_log_path: Path | None = None,
    content_profile=None,
) -> list[dict]:
    """B4: rejoin CREATE-bound action items into compound memories at natural granularity
    before B3 reconcile executes. Consumes B2's action-item list and emits a possibly-shorter
    list of the SAME shape — grouped creates replaced by composed creates, everything else
    (corroborate/contradict, unaddressed/singleton/rejected-group creates) passed through
    unchanged at its original position. B3 needs ZERO changes (see module note).

    See the internal design spec for the full contract; the guard
    numbers referenced below match that spec.

    ``content_profile`` is threaded from ``_run_extract_path`` (which already receives it and
    passes it to B3) so B4's content-safety check (``_b4_composed_content_is_safe`` ->
    ``_evaluate_create_content``) evaluates composed content under the SAME adaptive
    thresholds B3 will actually apply at persist time — B4 has a real production
    ``content_profile`` to thread through; it is not the profile-less default some earlier
    revisions of this docstring claimed (Sentinel, recall#884 re-review round 2).

    NOTE: ``decision_log_path=None`` is a total, silent opt-out of guard 3's traceability —
    a composition proceeds with zero provenance logged, the same as if the write had failed.
    Every real production call site (``_run_extract_path``) always supplies a concrete path,
    so this is reachable only for a caller that deliberately omits it.
    """
    create_indices = [i for i, item in enumerate(action_items) if item.get("action") == "create"]
    if not create_indices:
        return action_items  # guard 1 scope line: nothing to compose, no inference call

    creates = [action_items[i] for i in create_indices]
    n = len(creates)
    prompt = _build_rejoin_prompt(creates)
    budget = _estimate_rejoin_budget(prompt, n)

    # B4 compose bounded retry (A4) — see _B4_COMPOSE_MAX_ATTEMPTS docstring. Attempt 0's prompt
    # is byte-identical to the pre-retry single-shot prompt; only a failed attempt mutates it.
    raw_groups: "list[tuple[list, str, list, str | None]] | None" = None
    group_real_members: list[list[int]] | None = None
    last_reason = ""
    for attempt in range(_B4_COMPOSE_MAX_ATTEMPTS):
        attempt_prompt = prompt if attempt == 0 else (
            f"{prompt}\n\nYour previous response was rejected: {last_reason}. "
            "Fix this and respond again with valid JSON only."
        )
        request = {
            "prompt": attempt_prompt,
            "messages": [{"role": "user", "content": attempt_prompt}],
            "capabilities": [],
            "max_tokens": budget,
        }

        try:
            response = infer(request)
        except Exception as exc:
            last_reason = f"compose inference backend unavailable, exception: {exc!r}"
            continue

        parsed = _parse_llm_response(response)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("groups"), list):
            last_reason = "compose response unparseable or wrong group shape/schema"
            continue

        # Guard 3 (schema): a group entry with the WRONG TYPE for indices/content (e.g. a
        # string "0,1" instead of a list) is not an intentional unaddressed response — it
        # means the model's response as a whole doesn't conform to the contract. Guard 1
        # (never-drop) + duplicate detection: real (in-range, non-bool int) member indices are
        # computed for every group BEFORE deciding whether to drop any for containing an
        # invalid one — duplicate membership is scanned across every group's real members
        # regardless of whether that specific group later gets dropped, since a real index
        # appearing in two DIFFERENT proposed groups means the model's own bookkeeping is
        # incoherent even if one of those groups also happens to contain a hallucinated index
        # (fruit-confirmed, Sentinel: [0,1,999] + [0,2] let index 0's cross-group collision
        # through when the first group was filtered BEFORE the duplicate scan ran).
        raw_groups, group_real_members, reason = _b4_validate_compose_response(parsed, n)
        if reason is None:
            break
        last_reason = reason
    else:
        return _b4_fail_open(
            action_items, cluster_id,
            f"exhausted {_B4_COMPOSE_MAX_ATTEMPTS} attempts, last reason: {last_reason}",
        )

    # Malformed-supersedes telemetry (Sentinel, A3 re-review spec-author ruling, m_4adb43c0):
    # fires for EVERY group whose supersedes field was malformed, independent of what else
    # happens to that group (invalid-index drop, content-unsafe rejection, etc.) — the
    # detection-signal failure is worth recording on its own, before any other guard's fate is
    # decided for the group.
    for idxs, _content, _supersedes, malformed_reason in raw_groups:
        if malformed_reason is not None:
            _log_b4_group_rejection(
                decision_log_path, cluster_id, idxs, "malformed-supersedes", malformed_reason,
            )

    # NOW drop any group that contained an invalid/hallucinated index (per-group degradation —
    # its real members are not partially trusted under text written for a fabricated/different
    # membership set) or was empty to begin with.
    candidate_groups: list[tuple[list[int], str, list[int]]] = []
    for (idxs, content, supersedes, _malformed), real in zip(raw_groups, group_real_members):
        if not real or len(real) != len(idxs):
            _log_b4_group_rejection(
                decision_log_path, cluster_id, real or idxs, "invalid-index",
                "group contained an invalid or hallucinated member index",
            )
            continue
        candidate_groups.append((real, content, supersedes))

    # Guard 4 + composition: singleton groups (len 1) are pass-through, never rewritten, even
    # if the model proposed text for them. Multi-member groups compose only if their bounds
    # don't conflict (guard 4), their composed content is safe for B3's real content pipeline
    # (guard "content" below), and — when a decision-log path was given — provenance actually
    # persisted (guard 3 traceability below). Any of these failing is per-group degradation,
    # not the whole cluster.
    successful_by_first_pos: dict[int, dict] = {}
    composed_member_positions: set[int] = set()
    for idxs, content, _supersedes in candidate_groups:
        if len(idxs) < 2:
            continue

        # Guard "content": reject a composition that would silently vanish or degrade once it
        # reaches _apply_consolidation_result's real create branch — see
        # _b4_composed_content_is_safe for the full list of checks reused from B3. Members
        # fall back to individual pass-through.
        if not _b4_composed_content_is_safe(content, content_profile):
            _log_b4_group_rejection(
                decision_log_path, cluster_id, idxs, "content-unsafe",
                "composed content failed B3's shared create-content gate",
            )
            continue

        members = [creates[i] for i in idxs]
        bound = _group_temporal_bound(members)
        if bound is None:
            # guard 4 conflict: this group's members fall back to pass-through
            _log_b4_group_rejection(
                decision_log_path, cluster_id, idxs, "temporal-conflict",
                "members carry conflicting non-null valid_from/valid_until bounds",
            )
            continue
        categories = [m["category"] for m in members]
        category = Counter(categories).most_common(1)[0][0]
        # Sanitize member-supplied tag elements the SAME way _apply_consolidation_result's own
        # monolith-path tags do (scrub_text(str(t)) for t in tags if t) — a real B2 response can
        # carry a non-string tag element (fruit-confirmed, Sentinel: ["good", 7]), and a bare
        # sorted({...}) over a mixed-type set raises TypeError, crashing B4 entirely outside the
        # fail-open path. str() coercion first means the set is always string-only; sorted()
        # can never see a type it can't compare.
        tags = sorted({
            scrub_text(str(t)) for m in members for t in (m.get("tags") or []) if t
        })

        if decision_log_path is not None:
            logged_ok = _log_b4_compose_decision(
                decision_log_path, cluster_id, idxs, members, content,
            )
            if not logged_ok:
                # Guard 3 traceability: the decision log is v1's ONLY durable member
                # provenance (fruit-confirmed, Sentinel). A write failure means this
                # composition CANNOT be traced back to its members — reject it loudly rather
                # than persist an untraceable compound; members fall back to pass-through.
                logger.warning(
                    "B4_COMPOSE_FAIL_OPEN cluster=%s reason=%s member_indices=%s",
                    cluster_id, "decision-log provenance write failed", idxs,
                )
                continue

        composed = {
            "action": "create",
            "existing_id": None,
            "content": content,
            "category": category,
            "tags": tags,
            "source_turns": [],
            "contradiction_note": "",
            "valid_from": bound[0],
            "valid_until": bound[1],
        }
        first_original_pos = create_indices[idxs[0]]
        successful_by_first_pos[first_original_pos] = composed
        for i in idxs:
            composed_member_positions.add(create_indices[i])

    # Guard 6 (A3, supersession suppression — NOT a new numbered guard for THIS module note's
    # own counting scheme; "guard 6" is the A1 contract's name for this specific mechanism,
    # config f552875 section 4). A group's `supersedes` claim marks a candidate index this
    # group's content updates, corrects, or contradicts — same subject, different point in
    # time. Direction is NEVER trusted from the model: source_unit_id chronology
    # (_source_unit_id_order) is the sole arbiter, because the model can and does attribute the
    # claim to the wrong side (fruit: A2's negation-flip fixture flags the relationship from the
    # chronologically-earlier group's own entry). A claim only fires when BOTH sides of the pair
    # actually produced real output — a group already rejected by an earlier guard cannot be the
    # superseding OR superseded side; suppressing one side of a claim that never really resolved
    # to independent output would violate never-drop.
    member_to_group_first: dict[int, int] = {}
    group_supersedes_by_first: dict[int, list[int]] = {}
    group_is_singleton_by_first: dict[int, bool] = {}
    for idxs, _content, supersedes in candidate_groups:
        first = idxs[0]
        for i in idxs:
            member_to_group_first[i] = first
        group_supersedes_by_first[first] = supersedes
        group_is_singleton_by_first[first] = len(idxs) < 2

    def _surviving_position(creates_index: int) -> "int | None":
        """The original action_items position *creates_index*'s content actually appears at in
        the final output, or None if it never independently resolves to real output. For a
        SUCCESSFULLY composed multi-member group, EVERY member maps to the SAME composed output
        position (Sentinel, A3 re-review: "supersedes is defined over any candidate index, so
        first-member position cannot change the verdict" — a fix-confirmed bug: targeting a
        non-first member of an accepted [0,1] composition used to silently no-op while targeting
        member 0 suppressed the whole composed output, purely because of which original index
        the claim happened to name, even though both indices now refer to ONE logical memory)."""
        first = member_to_group_first.get(creates_index)
        if first is None:
            return None
        if group_is_singleton_by_first[first]:
            return create_indices[creates_index]  # singletons always survive at their own pos
        if create_indices[first] in successful_by_first_pos:
            # composed successfully — ALL members map to the ONE composed output position.
            return create_indices[first]
        # composed group was rejected by an earlier guard — every member reverts to its own
        # original, unmodified pass-through position.
        return create_indices[creates_index]

    def _group_claim_is_valid(first: int) -> bool:
        """Whether the group whose first member is *first* actually produced the composed
        content its `supersedes` claim was attached to. A claim from a REJECTED multi-member
        group must never fire, even though its individual members revert to pass-through —
        the composed sentence that CARRIED the claim never became real output; reverted
        members are unrelated atomic facts with no claim of their own (fruit-confirmed:
        without this check, a content-unsafe-rejected group's supersedes target was wrongly
        suppressed even though the rejected group itself fully vanished)."""
        if group_is_singleton_by_first[first]:
            return True  # a singleton's own fact IS its claim — nothing to have failed
        return create_indices[first] in successful_by_first_pos

    suppressed_positions: set[int] = set()
    seen_pairs: set[frozenset] = set()
    for idxs, _content, supersedes in candidate_groups:
        if not supersedes:
            continue
        representative = idxs[0]
        if not _group_claim_is_valid(representative):
            continue  # the claiming group's own composition never succeeded
        for target in supersedes:
            if target == representative:
                continue
            pair = frozenset((representative, target))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            rep_pos = _surviving_position(representative)
            target_pos = _surviving_position(target)
            if rep_pos is None or target_pos is None:
                continue  # one side never independently resolved to real output
            if rep_pos == target_pos:
                # Sentinel, A3 addendum: with every accepted-group member now mapping to the
                # SAME output position, a group's OWN supersedes claim against its own sibling
                # member (e.g. successful [0,1] with supersedes=[1] on member 0) would otherwise
                # try to suppress its sole composed output with itself — a self-reference, not a
                # genuine cross-entity supersession. Never-drop: skip it.
                continue

            rep_source = creates[representative].get("source_unit_id")
            target_source = creates[target].get("source_unit_id")
            order = _source_unit_id_order(rep_source, target_source)
            if order == 0:
                # A1 §7: same-entry siblings carry "no usable direction signal from
                # source_unit_id alone." A tie is not a coin flip — preserve both candidates and
                # record that a claim existed but couldn't be resolved (Sentinel, A3 re-review).
                _log_b4_group_rejection(
                    decision_log_path, cluster_id, [representative, target],
                    "ambiguous-supersession-order",
                    "supersedes claim between same-entry source_unit_ids has no usable "
                    "chronological direction — both candidates preserved",
                )
                continue
            superseded, superseding = (
                (representative, target) if order == -1 else (target, representative)
            )

            superseded_pos = _surviving_position(superseded)
            if superseded_pos is None or superseded_pos in suppressed_positions:
                continue
            logged_ok = _log_b4_supersede_decision(
                decision_log_path, cluster_id, superseded, superseding,
                creates[superseded].get("source_unit_id"),
                creates[superseding].get("source_unit_id"),
                creates[superseded]["content"], creates[superseding]["content"],
            )
            if not logged_ok:
                # Sentinel, A3 re-review + audit clarification: suppression is only truth-
                # preserving when the audit record is durable. A failed (or opted-out) write
                # means both candidates stay — never suppress on an untraceable decision.
                continue
            suppressed_positions.add(superseded_pos)

    output: list[dict] = []
    for pos, item in enumerate(action_items):
        if pos in suppressed_positions:
            continue  # guard 6: suppressed at creation, never enters output
        if item.get("action") != "create":
            output.append(item)
        elif pos in successful_by_first_pos:
            output.append(successful_by_first_pos[pos])
        elif pos in composed_member_positions:
            continue  # a non-first member of a successful group — already emitted
        else:
            output.append(item)  # unaddressed / singleton / rejected-group member
    return output


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

        cache_key = _cluster_cache_key(cluster)

        # Decomposed extract path (behind the flag): B1 (extract) -> B2 (action-decision) ->
        # B3 (reconcile). Runs BEFORE the monolithic response cache — its call shape is
        # per-unit/per-cluster, with no monolith-shaped cluster response to cache, and it must
        # not be skipped by a stale monolith cache entry. Reuses the SAME closed-over
        # existing_nodes/kn_path/decision_path/db/content_profile the monolith branch below
        # already uses — one consistent knowledge-state view regardless of which path ran.
        if _env_flag("SYNAPT_USE_EXTRACT"):
            if client is None:
                client = _get_consolidation_client()
                if client is None:
                    logger.error(
                        "No model backend available for extract-path consolidation"
                    )
                    return False
            cluster_result = _run_extract_path(
                cluster, cache_key, client, model, failures_path,
                existing_nodes, kn_path, decision_path, db, content_profile,
            )
            if cluster_result is None:
                return False
            result.nodes_created += cluster_result.nodes_created
            result.nodes_corroborated += cluster_result.nodes_corroborated
            result.nodes_contradicted += cluster_result.nodes_contradicted
            result.nodes_contested += cluster_result.nodes_contested
            return True

        cached_entry = response_cache.get(cache_key)

        if cached_entry:
            # Response was already applied on a previous run — the side
            # effects (append_node / update_node) are on disk.  Re-applying
            # would create duplicate nodes and double-count corroborations.
            logger.debug("Cache hit for cluster %s — skipping", cache_key)
            return True

        if client is None:
            client = _get_consolidation_client()
            if client is None:
                logger.error("No model backend available for consolidation")
                return False
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
        result.nodes_contested += cluster_result.nodes_contested

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
                    "Cluster %d/%d done: %d nodes created, %d corroborated, %d contested",
                    ci + 1, n_clusters,
                    result.nodes_created, result.nodes_corroborated, result.nodes_contested,
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
        if (
            result.nodes_created or result.nodes_corroborated
            or result.nodes_contradicted or result.nodes_contested
        ):
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
