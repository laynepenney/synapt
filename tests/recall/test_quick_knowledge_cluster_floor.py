"""TDD contract: recall_quick's concise mode emits ALL
knowledge_results before ever considering cluster_hits, with no floor or
cross-list score comparison. A handful of off-topic but high-confidence
knowledge rows can therefore fill the whole token budget and starve a
genuinely on-topic cluster that scores lower only because knowledge rows
carry a static confidence*specificity*knowledge_boost multiplier clusters
never get.

Reproduced live on a real gripspace index with the reported query
("origami token model design new tokenizer small model"): five off-topic
knowledge rows scored 27-44 (confidence-boosted, no relation to origami or
tokenizers) ran ahead of four clusters literally tagged "origami" scoring
~12.65 -- the origami clusters never appeared in the 500-token output.
This fixture reproduces the same SHAPE deterministically: several
off-topic, high-confidence knowledge rows sharing an incidental token with
the query, one on-topic cluster whose own FTS score is lower.
"""

from __future__ import annotations

from synapt.recall.core import TranscriptChunk, TranscriptIndex
from synapt.recall.storage import RecallDB


def _knowledge_results_for(index: TranscriptIndex, query: str) -> list[dict]:
    """Compute knowledge_results the way _global_lookup_fts actually does,
    so tests exercise the real caller contract instead of a stub."""
    return index._search_knowledge(
        query, 5, include_historical=False, knowledge_boost=2.0, emb_weight=1.0,
        query_entities=[], intent="general", after=None, before=None,
    )


def _off_topic_knowledge_nodes() -> list[dict]:
    # Five high-confidence rows sharing the incidental term "token" with the
    # query, none about origami or tokenizers -- the same shape as the live
    # repro's "gr spawn Codex foundation bridge" / "F-canonical" / etc rows.
    # Content is paragraph-length (matching real knowledge-node content) so
    # five of them alone are enough to consume the 500-token quick budget,
    # the same magnitude the live repro showed.
    topics = [
        (
            "kn-gr-spawn-bridge",
            "gr spawn Codex foundation bridge token budget note. Driver: Layne "
            "flagged two gaps when routing through Codex during Claude credit "
            "pressure, a token-preservation pattern for the fleet's spawn path "
            "and how session state carries across the bridge without loss.",
        ),
        (
            "kn-f-canonical",
            "F-canonical naming language token grammar elements. Introduced as "
            "a small typed declarative computing language for distributed "
            "agent coordination, a naming and addressing language with "
            "composition across nouns, verbs, and scoped resolution paths.",
        ),
        (
            "kn-memory-substrate",
            "Memory as substrate resource token cost reduction pointer. "
            "Memory and bookmarks should be git commits; knowledge node, "
            "memory file, MEMORY.md, and sticky reminder are different VIEWS "
            "into the same commit history, not duplicated content across them.",
        ),
        (
            "kn-200k-pref",
            "User prefers 200K context token model over the 1M context "
            "variant. Ironically the 1M context model seems to cause more "
            "forgetting in practice, likely because it does not focus as "
            "sharply on the immediate token window when the budget is larger.",
        ),
        (
            "kn-conversa-report",
            "Conversa team effectiveness report token usage summary. Four "
            "agents building a production app identified the same top five "
            "issues independently, spanning channel posting, response cost, "
            "and how each agent's token budget got spent across a session.",
        ),
    ]
    # Padded uniformly to ~160 tokens/block -- no SINGLE row exceeds the
    # 500-token budget alone (so this is not testing the separate "reject
    # an oversized first item" behavior), but three of the five together do
    # (3 * ~160 = ~480, close enough that a fourth breaks the loop and a
    # cluster's own ~32-token block can't fit in the ~20 tokens left over).
    # This isolates the RESERVATION specifically: without it, three rows
    # fit and starve the clusters; with it, only two rows fit, leaving
    # enough room for both cluster hits below to survive.
    _padding = (
        " Additional background context padding this knowledge node out to "
        "a realistic paragraph length, matching production knowledge nodes "
        "which typically run several sentences with source attribution and "
        "cross-references to the sessions and decisions that produced them, "
        "written out in enough detail to be genuinely useful on its own."
    )
    return [
        {
            "id": item_id,
            "content": content + _padding,
            "category": "decision",
            "confidence": 0.9,
            "source_sessions": [],
            "created_at": "2026-05-01T00:00:00+00:00",
            "updated_at": "2026-05-01T00:00:00+00:00",
            "status": "active",
            "tags": [],
        }
        for item_id, content in topics
    ]


def _index_with_starved_cluster_fixture(tmp_path) -> TranscriptIndex:
    """Several moderately-sized, off-topic knowledge rows cumulatively
    exhaust the 500-token budget (no single row is oversized on its own),
    so NEITHER cluster below (the on-topic origami one nor an off-topic
    decoy) gets any tokens at all -- matching the live repro, where the
    reported query returned five knowledge rows and zero cluster content.
    The decoy proves the floor doesn't just get lucky because only one
    cluster candidate exists."""
    db = RecallDB(tmp_path / "recall.db")
    db.save_knowledge_nodes(_off_topic_knowledge_nodes())

    chunks = [
        TranscriptChunk(
            id="s1:t0",
            session_id="sess-origami",
            timestamp="2026-07-10T09:00:00+00:00",
            turn_index=0,
            user_text="new tokenizer design using origami folds for a small model",
            assistant_text="origami tokenizer approach: hierarchical token folding",
        ),
        TranscriptChunk(
            id="s3:t0",
            session_id="sess-decoy",
            timestamp="2026-02-02T09:00:00+00:00",
            turn_index=0,
            user_text="generic token model design discussion, new small model",
            assistant_text="unrelated token model design notes, nothing origami here",
        ),
    ]
    db.save_chunks(chunks)
    db.save_clusters(
        [
            {
                "cluster_id": "clust-origami-tokenizer",
                "topic": "origami tokenizer small model design",
                "cluster_type": "topic",
                "session_ids": ["sess-origami"],
                "branch": None,
                "date_start": "2026-07-10T09:00:00+00:00",
                "date_end": "2026-07-10T09:00:00+00:00",
                "chunk_count": 1,
                "status": "active",
                "created_at": "2026-07-10T09:05:00+00:00",
                "updated_at": "2026-07-10T09:05:00+00:00",
            },
            {
                "cluster_id": "clust-decoy-token-model",
                "topic": "token model design new small model discussion",
                "cluster_type": "topic",
                "session_ids": ["sess-decoy"],
                "branch": None,
                "date_start": "2026-02-02T09:00:00+00:00",
                "date_end": "2026-02-02T09:00:00+00:00",
                "chunk_count": 1,
                "status": "active",
                "created_at": "2026-02-02T09:05:00+00:00",
                "updated_at": "2026-02-02T09:05:00+00:00",
            },
        ],
        [
            ("clust-origami-tokenizer", "s1:t0", "2026-07-10T09:05:00+00:00"),
            ("clust-decoy-token-model", "s3:t0", "2026-02-02T09:05:00+00:00"),
        ],
    )
    return TranscriptIndex([], use_embeddings=False, cache_dir=tmp_path, db=db)


QUERY = "origami token model design new tokenizer small model"


def test_concise_lookup_scores_confirm_the_starvation_shape(tmp_path) -> None:
    """Ground truth: the on-topic cluster's OWN score is lower than the
    off-topic knowledge rows' boosted scores -- so a naive fix that just
    sorts everything together by raw score, without a floor, would still
    bury it. Printed per the working-proof instruction."""
    index = _index_with_starved_cluster_fixture(tmp_path)

    knowledge_results = index._search_knowledge(
        QUERY, 5, include_historical=False, knowledge_boost=2.0, emb_weight=1.0,
        query_entities=[], intent="general", after=None, before=None,
    )
    cluster_hits = index._db.cluster_fts_search(QUERY, limit=10, include_archived=False)

    assert knowledge_results, "fixture must produce off-topic knowledge hits"
    assert cluster_hits, "fixture must produce the on-topic cluster hit"
    top_knowledge_score = knowledge_results[0]["score"]
    top_cluster_score = cluster_hits[0][1]
    assert top_knowledge_score > top_cluster_score, (
        "fixture design requires the off-topic knowledge score to beat the "
        "on-topic cluster score, matching the live repro's shape"
    )


def test_concise_lookup_does_not_starve_an_on_topic_cluster_behind_off_topic_knowledge(
    tmp_path,
) -> None:
    index = _index_with_starved_cluster_fixture(tmp_path)
    knowledge_results = _knowledge_results_for(index, QUERY)

    result = index._concise_lookup(QUERY, 5, 500, knowledge_results)

    assert "origami" in result.lower(), (
        "the on-topic origami cluster must survive the token budget even "
        "though every off-topic knowledge row individually outscores it"
    )


def test_concise_lookup_still_lists_on_topic_knowledge_first_unchanged(tmp_path) -> None:
    """Control: when the knowledge rows genuinely ARE the best match, the
    floor must not demote or reorder them -- this only gates the case where
    a lower-scoring, on-topic competitor is being starved."""
    db = RecallDB(tmp_path / "recall.db")
    db.save_knowledge_nodes(
        [
            {
                "id": "kn-origami-authoritative",
                "content": "origami tokenizer design: hierarchical token folding for small models",
                "category": "decision",
                "confidence": 0.95,
                "source_sessions": [],
                "created_at": "2026-08-01T00:00:00+00:00",
                "updated_at": "2026-08-01T00:00:00+00:00",
                "status": "active",
                "tags": [],
            }
        ]
    )
    chunks = [
        TranscriptChunk(
            id="s2:t0",
            session_id="sess-unrelated",
            timestamp="2026-03-01T09:00:00+00:00",
            turn_index=0,
            user_text="unrelated kubernetes deployment discussion",
            assistant_text="helm chart notes",
        ),
    ]
    db.save_chunks(chunks)
    db.save_clusters(
        [
            {
                "cluster_id": "clust-unrelated",
                "topic": "kubernetes deployment helm",
                "cluster_type": "topic",
                "session_ids": ["sess-unrelated"],
                "branch": None,
                "date_start": "2026-03-01T09:00:00+00:00",
                "date_end": "2026-03-01T09:00:00+00:00",
                "chunk_count": 1,
                "status": "active",
                "created_at": "2026-03-01T09:05:00+00:00",
                "updated_at": "2026-03-01T09:05:00+00:00",
            }
        ],
        [("clust-unrelated", "s2:t0", "2026-03-01T09:05:00+00:00")],
    )
    index = TranscriptIndex([], use_embeddings=False, cache_dir=tmp_path, db=db)
    knowledge_results = _knowledge_results_for(index, QUERY)

    result = index._concise_lookup(QUERY, 5, 500, knowledge_results)

    assert "origami" in result.lower()
    # The unrelated kubernetes cluster genuinely does not match this query
    # at all (zero term overlap) -- cluster_fts_search should not even
    # return it, so its absence here is not evidence of the floor at work.
    # Robustness over a raw char-offset: origami must appear in the FIRST
    # block (before the second "--- [" block header starts).
    blocks = result.split("--- [")
    assert len(blocks) >= 2, "expected at least one knowledge/cluster block"
    assert "origami" in blocks[1].lower(), (
        "on-topic knowledge must still appear as the first block, unchanged"
    )
