"""recall_code: composes the code symbol index (code_index.py) with ordinary
recall_search to answer natural-language questions about this repo's code and
what the team has said about it. No new index, no new store -- read-only
composition over two existing OSS primitives, plus an optional per-hit
annotation discovered through the same entry-point seam shape already used
by ``synapt.backends`` (see ``_model_router.py``).

Design history (2026-09-02, exploratory-then-hardened): the naive tokenizer
first tried fed literal English filler words to find_symbols, which ranks
exact > prefix > SUBSTRING (code_index.py). "catch" from a natural-language
question substring-matched unrelated cmd_catchup-family symbols and crowded
out the actually-named symbol within a fixed result budget. A stopword
filter and a global rank-then-truncate sort (collect every candidate across
every token first, THEN sort by match kind, THEN truncate) close that gap
structurally: a token whose own substring noise alone reaches max_symbols
can no longer crowd out a later token's exact hit by arrival order, which a
stopword blocklist alone cannot guarantee (it is necessarily incomplete).

No intent classifier: recall_code stays best-effort and surfaces HOW each
hit matched (exact/prefix/substring) and which query token produced it, so
the caller judges relevance itself -- a question with no real symbol in it
(e.g. "why is config push-and-resolve to main with no PR") still returns
code hits, but every one reads as substring/prefix noise on a generic word
rather than a confident false positive."""

from __future__ import annotations

import importlib.metadata
import logging
import re
from typing import Callable

from synapt.recall.code_index import find_symbols
from synapt.recall import server as recall_server

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    """
    a an the is are was were be been being do does did doesn doesnt don dont
    why how what when where who which but and or not no yes if then than
    this that these those it its of to in on at by for with without from
    as into onto up down out over under again further once here there all
    any both each few more most other some such only own same so too very
    can will just should now catch catches caught
    """.split()
)

_MATCH_KIND_RANK = {"exact": 0, "prefix": 1, "substring": 2}

# How many candidates each token contributes before the global sort -- must
# exceed max_symbols, or a token with enough of its own noise (e.g. "config"
# alone can return several substring hits) still crowds out a later token's
# exact hit within its own per-token slice before the global sort ever sees
# it.
_CANDIDATE_POOL_PER_TOKEN = 20

# --- Optional per-hit annotator seam ---
# A downstream layer may register a per-hit annotation callable via the
# 'synapt.annotators' entry point group -- same discovery shape as
# 'synapt.backends' in _model_router.py: loaded once, cached, and a missing
# or failing entry point degrades to no annotation rather than an error.
# Signature: annotator(repo_root, path, line_start, line_end) -> list[dict]

_ANNOTATOR_GROUP = "synapt.annotators"
_annotator_loaded: bool = False
_annotator: Callable[[str, str, int, int], list[dict]] | None = None


def _load_annotator() -> Callable[[str, str, int, int], list[dict]] | None:
    """Discover the optional per-hit annotator via the synapt.annotators
    entry-point group. Loaded once and cached for the process; an absent
    group, or an entry point that fails to load, leaves annotation off."""
    global _annotator_loaded, _annotator
    if _annotator_loaded:
        return _annotator
    _annotator_loaded = True
    for ep in importlib.metadata.entry_points(group=_ANNOTATOR_GROUP):
        try:
            _annotator = ep.load()
            logger.debug("Loaded annotator entry point: %s", ep.name)
            break
        except Exception:
            logger.debug(
                "Annotator entry point %r failed to load", ep.name, exc_info=True
            )
    return _annotator


def _identifier_tokens(query: str) -> list[str]:
    """Split a natural-language query into candidate symbol-name tokens.

    A phrase like "cold no-caller refresh" contains no literal symbol name
    (the real function might be cold_no_caller_refresh), so this also tries
    the whole query joined on underscores and on nothing, in addition to raw
    tokens -- the thinnest thing that could plausibly match without a real
    tokenizer or fuzzy matcher. English filler words are dropped so they
    don't crowd out real symbol names in the fixed-size result budget."""
    raw = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query)
    raw = [t for t in raw if t.lower() not in _STOPWORDS]
    joined_underscore = "_".join(re.findall(r"[A-Za-z]+", query))
    joined_none = "".join(re.findall(r"[A-Za-z]+", query)).lower()
    candidates = list(dict.fromkeys(raw + [joined_underscore, joined_none, query]))
    return [c for c in candidates if len(c) >= 3]


def _match_kind(symbol_name: str, token: str) -> str:
    """Mirror find_symbols' own rank (code_index.py) so the caller can see
    it, without touching the shared primitive: exact, prefix, or substring.
    find_symbols' WHERE clause guarantees every returned row contains
    `token` as a case-insensitive substring of `symbol_name`, so there is no
    fourth case -- these three are exhaustive over what it can return."""
    if symbol_name == token:
        return "exact"
    if symbol_name.lower().startswith(token.lower()):
        return "prefix"
    return "substring"


def recall_code(
    query: str,
    *,
    db_path: str,
    repo: str,
    repo_root: str,
    max_symbols: int = 5,
    max_chunks: int = 3,
) -> dict:
    """Answer a natural-language question about this repo's code plus what
    the team has said about it. Composes find_symbols (code index) with
    recall_search (transcript/channel/journal memory) and, through the
    optional synapt.annotators seam, a per-hit annotation.

    ``db_path``/``repo`` select the code index to query (see
    ``code_index.index_repo``); ``repo_root`` is the working tree an
    annotator would read from. No caller-global state -- every input is
    explicit so a caller (CLI, MCP tool, dashboard) can point this at any
    indexed repo.

    Returns:
        {
            "query": str,
            "symbols": [{name, kind, path, line_start, line_end, signature,
                         matched_token, match_kind, annotation?,
                         annotation_error?}],
            "has_code_hit": bool,
            "has_memory_hit": bool,
            "memories": str,
        }

    ``annotation`` is present only when an annotator is registered and
    succeeds; its absence (no annotator registered) is silent -- no error,
    no "annotation_error" key. An annotator import/call failure sets
    "annotation_error" instead of raising, since annotation is enrichment,
    not the answer.
    """
    seen: set[tuple[str, str, int]] = set()
    candidates: list[dict] = []
    for token in _identifier_tokens(query):
        for hit in find_symbols(db_path, token, repo=repo, limit=_CANDIDATE_POOL_PER_TOKEN):
            key = (hit["path"], hit["name"], hit["line_start"])
            if key in seen:
                continue
            seen.add(key)
            hit["matched_token"] = token
            hit["match_kind"] = _match_kind(hit["name"], token)
            candidates.append(hit)
    candidates.sort(key=lambda h: (_MATCH_KIND_RANK[h["match_kind"]], h["name"]))
    symbol_hits = candidates[:max_symbols]

    annotator = _load_annotator()
    if annotator is not None:
        for hit in symbol_hits:
            try:
                hit["annotation"] = annotator(
                    repo_root, hit["path"], hit["line_start"], hit["line_end"]
                )
            except Exception as exc:  # noqa: BLE001 - enrichment, never the answer
                hit["annotation_error"] = str(exc)

    memory_text = recall_server.recall_search(query, max_chunks=max_chunks)
    memory_hit = "No results found." not in memory_text

    return {
        "query": query,
        "symbols": symbol_hits,
        "has_code_hit": bool(symbol_hits),
        "has_memory_hit": memory_hit,
        "memories": memory_text,
    }
